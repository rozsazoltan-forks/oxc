use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
};

use fast_glob::glob_match;
use ignore::gitignore::Gitignore;
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::instrument;

use oxc_config::{all_paths_have_vcs_boundary, configure_walk_builder};
use oxc_diagnostics::{DiagnosticSender, DiagnosticService, OxcDiagnostic};

use super::resolve::{build_global_ignore_matchers, is_ignored, resolve_file_scope_config};
#[cfg(feature = "napi")]
use crate::core::JsConfigLoaderCb;
use crate::core::{
    ConfigResolver, DiscoveryCtx, FormatStrategy, ResolveOutcome, classify_file_kind,
};

/// Orchestrates file discovery with nested config and ignore handling.
///
/// Constructed from CLI path arguments, which are classified into
/// target paths, glob patterns, and exclude patterns.
///
/// # Config resolution
/// Each file is formatted by its nearest config (auto-discovered upward).
/// Directories containing a config file form a scope boundary.
/// There is no inheritance between parent and child scopes.
///
/// When `--config` or `--disable-nested-config` is explicitly given, nested detection is disabled entirely.
///
/// # Walk phases
/// - Phase 1: Classify CLI positional PATHs into directory targets, file targets, and globs
/// - Phase 2: File targets are processed directly (no walk)
///   - Scope is resolved by walking `file.parent().ancestors()` and probing each
///     dir via `get_or_load_direct_config` (shared cache). The first hit wins;
///     reaching `root_config_dir` returns the pre-built root resolver without
///     re-loading.
///   - This helps with performance when many file targets are specified like with `husky`
///     (the shared cache deduplicates loads across siblings).
/// - Phase 3: Directory targets are walked via a single parallel walk
///   - Discovery is **entry-based**: `filter_entry` lets config-looking files
///     pass file-level global ignore so `visit()` can register them in
///     `scope_by_dir` (the walk-wide shared scope map).
///   - Non-config file `visit()` resolves scope by ancestor lookup against
///     `scope_by_dir` + the visitor-local `scope_cache`, with a race-rescue
///     `get_or_load_direct_config` probe across `visited` (closer-first) so
///     `nearest-config-wins` holds even when a closer config has not yet been
///     registered by another visitor.
///   - All loads (JSON / JSONC / JS / Vite) go through `config_load_cache`
///     (`OnceLock` per dir) so each directory's config is loaded **at most
///     once walk-wide**, including across Phase 2 / Phase 3.
///
/// # Ignore model
/// Three layers, checked in `filter_entry()` and `visit()`.
///
/// - (1) Hardcoded skips: `.git`, `.svn`, `node_modules`, etc
///   - Always skipped in `filter_entry()`, cannot be overridden by ignore files or patterns
/// - (2) Global ignores: `.prettierignore`, `--ignore-path`, CLI `"!path"`
///   - Block both directories and files across all scopes.
///   - **Exception**: `filter_entry` lets config-looking files (`.oxfmtrc.json`
///     etc.) bypass file-level global ignore so discovery can register their
///     scope. `visit()` re-applies global ignore to those entries before
///     deciding format eligibility.
/// - (3) Scope-local `ignorePatterns`: each scope's config can define patterns
///   - Applied per-file in `visit()` against the file's resolved scope. Inner
///     scopes' `ignorePatterns` correctly override outer ones because each
///     file resolves to its nearest config.
///
/// # Atomicity on broken nested config
///
/// Nested configs are loaded lazily during the walk. When one fails to parse,
/// the walk reports the error via `fatal_error` and stops, but format workers
/// run concurrently and may have already written files in unaffected scopes
/// to disk. Format is idempotent — the recommended recovery is to fix the
/// failing config and re-run. Buffering writes until walk completion would
/// give all-or-nothing semantics but at the cost of holding all formatted
/// outputs in memory, which doesn't scale to very large monorepos.
pub struct ScopedWalker {
    cwd: PathBuf,
    paths: Vec<PathBuf>,
    glob_patterns: Vec<String>,
    exclude_patterns: Vec<String>,
}

impl ScopedWalker {
    /// Create a new `ScopedWalker` by classifying CLI path arguments.
    ///
    /// Paths are split into target paths, glob patterns, and exclude patterns (`!` prefix).
    pub fn new(cwd: PathBuf, paths: &[PathBuf]) -> Self {
        let mut target_paths = vec![];
        let mut glob_patterns = vec![];
        let mut exclude_patterns = vec![];

        for path in paths {
            let path_str = path.to_string_lossy();

            // Instead of `oxlint`'s `--ignore-pattern=PAT`,
            // `oxfmt` supports `!` prefix in paths like Prettier.
            if path_str.starts_with('!') {
                exclude_patterns.push(path_str.to_string());
                continue;
            }

            // Normalize `./` prefix (and any consecutive slashes, e.g. `.//src/app.js`)
            let normalized = if let Some(stripped) = path_str.strip_prefix("./") {
                stripped.trim_start_matches('/')
            } else {
                &path_str
            };

            if is_glob_pattern(normalized, &cwd) {
                glob_patterns.push(normalized.to_string());
                continue;
            }

            let full_path = if path.is_absolute() {
                path.clone()
            } else if normalized == "." {
                // NOTE: `.` and cwd behave differently, need to normalize
                cwd.clone()
            } else {
                cwd.join(normalized)
            };
            target_paths.push(full_path);
        }

        Self { cwd, paths: target_paths, glob_patterns, exclude_patterns }
    }

    /// Run the walk across all scopes.
    /// And stream file to be formatted with its resolved config via the shared channel.
    ///
    /// Returns `Ok(true)` if any valid config was used.
    #[instrument(level = "debug", name = "oxfmt::walk::run", skip_all)]
    pub fn run(
        &self,
        root_config: ConfigResolver,
        ignore_paths: &[PathBuf],
        with_node_modules: bool,
        detect_nested: bool,
        editorconfig_path: Option<&Path>,
        #[cfg(feature = "napi")] js_config_loader: Option<&JsConfigLoaderCb>,
        sender: &mpsc::Sender<FormatStrategy>,
        tx_error: &DiagnosticSender,
    ) -> Result<bool, String> {
        let root_config_resolver = Arc::new(root_config);

        // Global ignores: .prettierignore, --ignore-path, CLI `!` patterns
        let ignore_file_matchers: Arc<[Gitignore]> = Arc::from(build_global_ignore_matchers(
            &self.cwd,
            &self.exclude_patterns,
            ignore_paths,
        )?);

        let mut any_config_used = root_config_resolver.config_dir().is_some();

        // Phase 1: Classify targets into directories (walk) and files (direct)
        let (walk_targets, file_targets) = {
            let mut initial_targets: FxHashSet<PathBuf> = self.paths.iter().cloned().collect();

            // When glob patterns exist, walk from cwd to find matching files during traversal.
            // Concrete file paths are still added individually as base paths.
            if !self.glob_patterns.is_empty() {
                initial_targets.insert(self.cwd.clone());
            }

            // Default to `cwd` if no positive paths were specified.
            // Exclude patterns alone should still walk, but unmatched globs should not.
            if initial_targets.is_empty() && self.glob_patterns.is_empty() {
                initial_targets.insert(self.cwd.clone());
            }

            let mut dirs = vec![];
            let mut files = vec![];
            for path in &initial_targets {
                // Base paths passed to `WalkBuilder` are not filtered by `filter_entry()`,
                // so we need to filter them here before passing to the walker.
                // This is needed for cases like `husky`, may specify ignored paths as staged files.
                // NOTE: Git ignored paths are not filtered here.
                // But it's OK because in cases like `husky`, they are never staged.
                if is_ignored(&ignore_file_matchers, path, path.is_dir(), true) {
                    continue;
                }

                if path.is_dir() {
                    dirs.push(path.clone());
                } else {
                    files.push(path.clone());
                }
            }
            (dirs, files)
        };

        // Walk-wide shared discovery state. Cloning `DiscoveryCtx` is cheap
        // (each field is `Arc` / `Copy`); the underlying caches and signals
        // are shared across Phase 2, Phase 3, and across all parallel visitors.
        let discovery_ctx = DiscoveryCtx::new(
            editorconfig_path.map(Arc::from),
            #[cfg(feature = "napi")]
            js_config_loader.cloned(),
        );
        let fatal_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // Phase 2: Process file targets directly (no walk needed).
        let mut directly_processed: FxHashSet<PathBuf> = FxHashSet::default();
        if !file_targets.is_empty() {
            let mut scope_cache: FxHashMap<&Path, Arc<ConfigResolver>> = FxHashMap::default();
            for file in &file_targets {
                // Skip non-existent files (WalkBuilder naturally skips these via error entries)
                if !file.is_file() {
                    continue;
                }

                let file_config = if detect_nested {
                    let parent = file.parent().unwrap();
                    if !scope_cache.contains_key(parent) {
                        let resolved =
                            resolve_file_scope_config(file, &root_config_resolver, &discovery_ctx)?;
                        scope_cache.insert(parent, resolved);
                    }
                    Arc::clone(&scope_cache[parent])
                } else {
                    Arc::clone(&root_config_resolver)
                };

                if file_config.is_path_ignored(file, false) {
                    continue;
                }

                let Some(kind) = classify_file_kind(Arc::from(file.as_path())) else {
                    continue;
                };
                let strategy = match file_config.resolve(kind) {
                    Ok(ResolveOutcome::Format(strategy)) => strategy,
                    Ok(ResolveOutcome::MissingPlugin(_)) => continue,
                    Err(err) => {
                        report_resolve_error(tx_error, &self.cwd, file, err);
                        continue;
                    }
                };

                directly_processed.insert(file.clone());
                if sender.send(strategy).is_err() {
                    break;
                }
            }
        }

        // Phase 3: Walk directory targets.
        let directly_processed: Arc<FxHashSet<PathBuf>> = Arc::new(directly_processed);

        // Build the glob matcher once for walk-time filtering.
        // When glob patterns exist, files are matched against them during `visit()`.
        // When no globs, `visit()` has zero overhead.
        let glob_matcher = (!self.glob_patterns.is_empty()).then(|| {
            Arc::new(GlobMatcher::new(self.cwd.clone(), self.glob_patterns.clone(), &walk_targets))
        });

        let has_vcs_boundary = all_paths_have_vcs_boundary(&walk_targets, &self.cwd);
        let walk_target_roots: Arc<[PathBuf]> = Arc::from(walk_targets.clone());

        walk_and_stream(
            &self.cwd,
            &walk_targets,
            has_vcs_boundary,
            Arc::clone(&ignore_file_matchers),
            with_node_modules,
            glob_matcher.as_ref(),
            &root_config_resolver,
            &directly_processed,
            discovery_ctx.clone(),
            detect_nested,
            walk_target_roots,
            Arc::clone(&fatal_error),
            sender,
            tx_error,
        );

        // Surface any fatal error encountered inside the parallel walk.
        let fatal = fatal_error.lock().expect("fatal_error mutex poisoned").take();
        if let Some(err) = fatal {
            return Err(err);
        }

        if discovery_ctx.config_found() {
            any_config_used = true;
        }

        Ok(any_config_used)
    }
}

// ---

/// Check if a path string looks like a glob pattern.
/// Glob-like characters are also valid path characters on some environments.
/// If the path actually exists on disk, it is treated as a concrete path.
/// e.g. `{config}.js`, `[id].tsx`
fn is_glob_pattern(s: &str, cwd: &Path) -> bool {
    let has_glob_chars = s.contains('*') || s.contains('?') || s.contains('[') || s.contains('{');
    has_glob_chars && !cwd.join(s).exists()
}

/// Matches file paths against glob patterns during walk.
///
/// When glob patterns are specified via CLI args,
/// files are matched against them during the walk's `visit()` callback.
///
/// Uses `fast_glob::glob_match` instead of `ignore::Overrides` because
/// overrides have the highest priority in the `ignore` crate and would bypass `.gitignore` rules.
///
/// Also tracks concrete target paths (non-glob) because when globs are present,
/// cwd is added as a base path, which means concrete paths can be visited twice.
/// (as direct base paths and during the cwd walk)
/// This struct handles both acceptance and dedup of those paths via `matches()`.
struct GlobMatcher {
    /// cwd for computing relative paths for glob matching.
    cwd: PathBuf,
    /// Normalized glob pattern strings for matching via `fast_glob::glob_match`.
    glob_patterns: Vec<String>,
    /// Concrete target paths (absolute) specified via CLI.
    /// These are always accepted even when glob filtering is active.
    concrete_paths: FxHashSet<PathBuf>,
    /// Tracks seen concrete paths to avoid duplicates (visited both as
    /// direct base paths and via cwd walk).
    seen: Mutex<FxHashSet<PathBuf>>,
}

impl GlobMatcher {
    fn new(cwd: PathBuf, glob_patterns: Vec<String>, target_paths: &[PathBuf]) -> Self {
        // Normalize glob patterns: patterns without `/` are prefixed with `**/`
        // to match at any depth (gitignore/prettier semantics).
        // e.g., `*.js` → `**/*.js`, `foo/**/*.js` stays as-is.
        let glob_patterns = glob_patterns
            .into_iter()
            .map(|pat| if pat.contains('/') { pat } else { format!("**/{pat}") })
            .collect();
        // Store concrete paths (excluding cwd itself) for dedup and acceptance.
        let concrete_paths = target_paths.iter().filter(|p| p.as_path() != cwd).cloned().collect();
        Self { cwd, glob_patterns, concrete_paths, seen: Mutex::new(FxHashSet::default()) }
    }

    /// Returns `true` if the path matches any glob pattern or is a concrete target path.
    /// Concrete paths are deduplicated (they can appear both as direct base paths and via cwd walk).
    fn matches(&self, path: &Path) -> bool {
        // Accept concrete paths (explicitly specified via CLI), with dedup
        if self.concrete_paths.contains(path) {
            return self.seen.lock().unwrap().insert(path.to_path_buf());
        }

        let relative = path.strip_prefix(&self.cwd).unwrap_or(path).to_string_lossy();
        self.glob_patterns.iter().any(|pattern| glob_match(pattern, relative.as_ref()))
    }
}

// ---

/// Build a Walk, stream entries to the shared channel.
#[expect(clippy::needless_pass_by_value)] // Arcs are moved into closures/structs
fn walk_and_stream(
    cwd: &Path,
    target_paths: &[PathBuf],
    has_vcs_boundary: bool,
    ignore_file_matchers: Arc<[Gitignore]>,
    with_node_modules: bool,
    glob_matcher: Option<&Arc<GlobMatcher>>,
    root_config_resolver: &Arc<ConfigResolver>,
    directly_processed: &Arc<FxHashSet<PathBuf>>,
    discovery_ctx: DiscoveryCtx,
    detect_nested: bool,
    walk_target_roots: Arc<[PathBuf]>,
    fatal_error: Arc<Mutex<Option<String>>>,
    sender: &mpsc::Sender<FormatStrategy>,
    tx_error: &DiagnosticSender,
) {
    let Some(first_path) = target_paths.first() else {
        return;
    };

    let mut inner = ignore::WalkBuilder::new(first_path);
    for path in target_paths.iter().skip(1) {
        inner.add(path);
    }

    let filter_global = Arc::clone(&ignore_file_matchers);
    let filter_discovery = discovery_ctx.discovery;
    inner.filter_entry(move |entry| {
        let Some(file_type) = entry.file_type() else {
            return false;
        };
        let is_dir = file_type.is_dir();

        if is_dir && is_walk_excluded_dir(entry, &filter_global, with_node_modules) {
            return false;
        }
        // File-level global ignores apply, EXCEPT for config-looking files —
        // those must reach `visit()` so entry-based discovery can register
        // them in `scope_by_dir`. Whether they get formatted is decided
        // separately in `visit()` by re-checking global ignore.
        if !is_dir
            && filter_discovery.discover_config_file(entry.path()).is_none()
            && is_ignored(&filter_global, entry.path(), false, false)
        {
            return false;
        }

        // Scope-local `ignorePatterns` are checked per-file in `visit()`
        // against the file's resolved scope, which correctly handles the
        // "inner config wins over outer" rule.
        //
        // Glob pattern matching is also done per-file in `visit()` since
        // patterns like `**/*.js` cannot reliably skip directories.
        true
    });

    let mut builder = WalkVisitorBuilder {
        cwd: Arc::from(cwd),
        sender: sender.clone(),
        tx_error: tx_error.clone(),
        root_config_resolver: Arc::clone(root_config_resolver),
        glob_matcher: glob_matcher.cloned(),
        directly_processed: Arc::clone(directly_processed),
        discovery_ctx,
        detect_nested,
        walk_target_roots,
        fatal_error,
        filter_global: Arc::clone(&ignore_file_matchers),
    };

    configure_oxfmt_walk_builder(&mut inner, has_vcs_boundary).build_parallel().visit(&mut builder);
}

/// Wrap [`oxc_config::configure_walk_builder`] with Oxfmt-specific options.
///
/// Git-related settings come from the shared helper to align with Oxlint.
/// Prettier only reads `.gitignore` in the cwd and does not respect `.git/info/exclude`.
fn configure_oxfmt_walk_builder(
    builder: &mut ignore::WalkBuilder,
    has_vcs_boundary: bool,
) -> &mut ignore::WalkBuilder {
    configure_walk_builder(builder, has_vcs_boundary)
        // Do not follow symlinks like Prettier does.
        // See https://github.com/prettier/prettier/pull/14627
        .follow_links(false)
        // Use the same thread count as rayon (controlled by `--threads`)
        .threads(rayon::current_num_threads())
}

struct WalkVisitorBuilder {
    cwd: Arc<Path>,
    sender: mpsc::Sender<FormatStrategy>,
    tx_error: DiagnosticSender,
    root_config_resolver: Arc<ConfigResolver>,
    glob_matcher: Option<Arc<GlobMatcher>>,
    directly_processed: Arc<FxHashSet<PathBuf>>,
    discovery_ctx: DiscoveryCtx,
    detect_nested: bool,
    walk_target_roots: Arc<[PathBuf]>,
    fatal_error: Arc<Mutex<Option<String>>>,
    /// Needed in `visit()` to gate format eligibility for config-looking files
    /// that bypass `filter_entry`.
    filter_global: Arc<[Gitignore]>,
}

impl<'s> ignore::ParallelVisitorBuilder<'s> for WalkVisitorBuilder {
    fn build(&mut self) -> Box<dyn ignore::ParallelVisitor + 's> {
        Box::new(WalkVisitor {
            cwd: Arc::clone(&self.cwd),
            sender: self.sender.clone(),
            tx_error: self.tx_error.clone(),
            root_config_resolver: Arc::clone(&self.root_config_resolver),
            glob_matcher: self.glob_matcher.clone(),
            directly_processed: Arc::clone(&self.directly_processed),
            discovery_ctx: self.discovery_ctx.clone(),
            detect_nested: self.detect_nested,
            walk_target_roots: Arc::clone(&self.walk_target_roots),
            fatal_error: Arc::clone(&self.fatal_error),
            filter_global: Arc::clone(&self.filter_global),
            scope_cache: FxHashMap::default(),
        })
    }
}

struct WalkVisitor {
    cwd: Arc<Path>,
    sender: mpsc::Sender<FormatStrategy>,
    tx_error: DiagnosticSender,
    root_config_resolver: Arc<ConfigResolver>,
    glob_matcher: Option<Arc<GlobMatcher>>,
    directly_processed: Arc<FxHashSet<PathBuf>>,
    discovery_ctx: DiscoveryCtx,
    detect_nested: bool,
    walk_target_roots: Arc<[PathBuf]>,
    fatal_error: Arc<Mutex<Option<String>>>,
    filter_global: Arc<[Gitignore]>,
    /// Visitor-local cache: parent dir → (resolved scope, parent_ignored flag).
    /// Entries are direct-probe-confirmed; under static-FS they stay valid
    /// for this visitor's lifetime.
    scope_cache: FxHashMap<PathBuf, (Arc<ConfigResolver>, bool)>,
}

impl WalkVisitor {
    /// Record the first fatal error seen by any visitor.
    fn record_fatal(&self, err: String) {
        let mut guard = self.fatal_error.lock().expect("fatal_error mutex poisoned");
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    /// Resolve and cache `parent`'s scope.
    ///
    /// Phase 1 walks `parent.ancestors()`, accumulating dirs without cache hit
    /// into `visited`. Phase 2 race-probes every dir in `visited` (closer-first)
    /// — even when Phase 1 hit an outer ancestor, this protects `nearest-config-wins`
    /// against parallel visitors that may register a closer config concurrently.
    fn ensure_scope_cached(&mut self, parent: &Path) -> Result<(), String> {
        if self.scope_cache.contains_key(parent) {
            return Ok(());
        }

        // detect_nested off → root scope only.
        if !self.detect_nested {
            let parent_ignored = self.root_config_resolver.is_path_ignored(parent, true);
            self.scope_cache.insert(
                parent.to_path_buf(),
                (Arc::clone(&self.root_config_resolver), parent_ignored),
            );
            return Ok(());
        }

        let probe_root: Option<&Path> = self
            .walk_target_roots
            .iter()
            .map(PathBuf::as_path)
            .filter(|t| parent.starts_with(t))
            .max_by_key(|t| t.components().count());
        let root_config_dir = self.root_config_resolver.config_dir();

        // Phase 1: cheap ancestor lookup (no probe).
        let mut visited: Vec<PathBuf> = vec![];
        let mut hit_via_lookup: Option<Arc<ConfigResolver>> = None;

        'lookup: for dir in parent.ancestors() {
            if let Some((r, _)) = self.scope_cache.get(dir) {
                hit_via_lookup = Some(Arc::clone(r));
                break 'lookup;
            }

            if Some(dir) == root_config_dir {
                hit_via_lookup = Some(Arc::clone(&self.root_config_resolver));
                break 'lookup;
            }

            let hit = {
                let guard =
                    self.discovery_ctx.scope_by_dir.read().expect("scope_by_dir rwlock poisoned");
                guard.get(dir).cloned()
            };
            if let Some(r) = hit {
                hit_via_lookup = Some(r);
                break 'lookup;
            }

            visited.push(dir.to_path_buf());
            if Some(dir) == probe_root {
                break;
            }
        }

        // Phase 2: race-rescue probe across `visited`, closer-first.
        let mut found_closer: Option<(PathBuf, Arc<ConfigResolver>)> = None;
        for dir in &visited {
            if let Some(loaded) = self.discovery_ctx.probe_dir(dir)? {
                found_closer = Some((dir.clone(), loaded));
                break;
            }
        }

        let (resolved_scope_dir, resolver) = match (found_closer, hit_via_lookup) {
            (Some((dir, loaded)), _) => (Some(dir), loaded),
            (None, Some(r)) => (None, r),
            (None, None) => (None, Arc::clone(&self.root_config_resolver)),
        };

        // Cache: (1) the dir that owns the resolved scope, (2) negative-cached
        // probed dirs, (3) `parent` itself if not yet covered.
        if let Some(scope_dir) = resolved_scope_dir
            && !self.scope_cache.contains_key(&scope_dir)
        {
            let parent_ignored = resolver.is_path_ignored(&scope_dir, true);
            self.scope_cache.insert(scope_dir, (Arc::clone(&resolver), parent_ignored));
        }
        for dir in visited {
            if self.scope_cache.contains_key(&dir) {
                continue;
            }
            let parent_ignored = resolver.is_path_ignored(&dir, true);
            self.scope_cache.insert(dir, (Arc::clone(&resolver), parent_ignored));
        }
        if !self.scope_cache.contains_key(parent) {
            let parent_ignored = resolver.is_path_ignored(parent, true);
            self.scope_cache.insert(parent.to_path_buf(), (resolver, parent_ignored));
        }

        Ok(())
    }
}

impl ignore::ParallelVisitor for WalkVisitor {
    fn visit(&mut self, entry: Result<ignore::DirEntry, ignore::Error>) -> ignore::WalkState {
        let entry = match entry {
            Ok(e) => e,
            Err(_err) => return ignore::WalkState::Skip,
        };

        let Some(file_type) = entry.file_type() else {
            return ignore::WalkState::Continue;
        };
        if file_type.is_dir() {
            // No per-dir probe in the discovery path — entry-based discovery
            // happens via `visit(config-looking file)` below.
            return ignore::WalkState::Continue;
        }

        // Skip non-regular entries (symlinks of any kind, sockets, etc.) to
        // match the walker's `follow_links(false)` behavior.
        #[expect(clippy::filetype_is_file)]
        if !file_type.is_file() {
            return ignore::WalkState::Continue;
        }

        let path = entry.into_path();

        // Skip files already processed as direct file targets.
        if self.directly_processed.contains(&path) {
            return ignore::WalkState::Continue;
        }

        let parent = path.parent().expect("walk yields absolute paths");

        // Identify config-looking files. Needed regardless of `detect_nested`
        // because (B) re-applies global ignore for them: `filter_entry`
        // exempts config files from file-level global ignore so discovery
        // can see them, and (B) restores that filter for format eligibility.
        let is_config_file = self.discovery_ctx.discovery.discover_config_file(&path).is_some();

        // (A) Discovery: register the parent dir's scope (nested-detection only).
        if is_config_file
            && self.detect_nested
            && let Err(err) = self.discovery_ctx.probe_dir(parent)
        {
            self.record_fatal(err);
            return ignore::WalkState::Quit;
        }

        // (B) Re-apply file-level global ignore for config-looking files.
        if is_config_file && is_ignored(&self.filter_global, &path, false, false) {
            return ignore::WalkState::Continue;
        }

        // (C) Resolve scope for this file (cached per parent directory).
        if let Err(err) = self.ensure_scope_cached(parent) {
            self.record_fatal(err);
            return ignore::WalkState::Quit;
        }
        let (resolver, parent_ignored) = &self.scope_cache[parent];

        // Scope-local `ignorePatterns` check.
        //
        // Two-level: parent dir (cached) catches directory patterns like
        // `lib`; file-level catches patterns like `temp.js`.
        if *parent_ignored || resolver.is_path_ignored(&path, false) {
            return ignore::WalkState::Continue;
        }

        // Glob filter (when active).
        if let Some(glob_matcher) = &self.glob_matcher
            && !glob_matcher.matches(&path)
        {
            return ignore::WalkState::Continue;
        }

        // Tier 1 = `.js`, `.tsx`, etc: JS/TS files supported by `oxc_formatter`
        // Tier 2 = `.toml`, etc: Some files supported by `oxfmt` directly
        // Tier 3 = `.html`, `.json`, etc: Other files supported by Prettier
        // (Tier 4 = `.astro`, `.svelte`, etc: Prettier plugins)
        // Anything else is silently skipped.
        let path: Arc<Path> = Arc::from(path);
        let Some(kind) = classify_file_kind(Arc::clone(&path)) else {
            return ignore::WalkState::Continue;
        };
        let strategy = match resolver.resolve(kind) {
            Ok(ResolveOutcome::Format(strategy)) => strategy,
            Ok(ResolveOutcome::MissingPlugin(_)) => {
                return ignore::WalkState::Continue;
            }
            Err(err) => {
                report_resolve_error(&self.tx_error, &self.cwd, &path, err);
                return ignore::WalkState::Continue;
            }
        };

        if self.sender.send(strategy).is_err() {
            return ignore::WalkState::Quit;
        }

        ignore::WalkState::Continue
    }
}

// ---

/// Report a per-file config resolve error via the diagnostic channel.
fn report_resolve_error(tx_error: &DiagnosticSender, cwd: &Path, path: &Path, err: String) {
    let diagnostics = DiagnosticService::wrap_diagnostics(
        cwd,
        path,
        "",
        vec![
            OxcDiagnostic::error(format!("Invalid resolved configuration for {}", path.display()))
                .with_help(err),
        ],
    );
    let _ = tx_error.send(diagnostics);
}

/// Check if a directory should be excluded from walking.
///
/// Skips VCS directories (`.git`, `.svn`, etc.), `node_modules` (by default),
/// and directories matched by global ignore files (`.prettierignore`, `--ignore-path`, CLI `!`).
fn is_walk_excluded_dir(
    entry: &ignore::DirEntry,
    global_matchers: &[Gitignore],
    with_node_modules: bool,
) -> bool {
    is_ignored_dir(entry.file_name(), with_node_modules)
        || is_ignored(global_matchers, entry.path(), true, false)
}

/// Check if a directory should be skipped during walking.
/// VCS internal directories are always skipped, and `node_modules` is skipped by default.
/// We set `.hidden(false)` on `WalkBuilder` to include hidden files,
/// but still skip these specific directories (matching Prettier's behavior).
/// <https://prettier.io/docs/ignore#ignoring-files-prettierignore>
fn is_ignored_dir(dir_name: &OsStr, with_node_modules: bool) -> bool {
    dir_name == ".git"
        || dir_name == ".jj"
        || dir_name == ".sl"
        || dir_name == ".svn"
        || dir_name == ".hg"
        || (!with_node_modules && dir_name == "node_modules")
}
