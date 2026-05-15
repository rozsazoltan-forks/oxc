use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};

use editorconfig_parser::EditorConfig;
use rustc_hash::FxHashMap;

use oxc_config::{ConfigConflict, ConfigDiscovery, DiscoveredConfigFile};

#[cfg(feature = "napi")]
use super::js_config::JsConfigLoaderCb;
use super::{
    ConfigResolver, build_resolver_from_discovered, config_discovery,
    editorconfig::load_editorconfig,
};

/// Find the unique config file directly inside `dir` using a single `read_dir`.
///
/// Unlike `ConfigDiscovery::find_unique_config_in_directory` which calls
/// `is_file()` per candidate name, this issues one `read_dir` and matches
/// entry names — no extra `stat` syscalls.
///
/// Only regular files are considered: directories and symlinks (regardless of
/// target) are skipped. This matches the walker's `follow_links(false)`
/// behavior so config discovery and file traversal stay consistent.
///
/// Takes a borrowed `ConfigDiscovery` so the caller's cached instance is
/// reused — building one calls `vp_version()` which acquires Rust's global
/// `ENV_LOCK`, serializing parallel callers on a hot path.
///
/// Returns `Ok(None)` when `dir` is unreadable; the caller can decide whether
/// that warrants a diagnostic.
fn find_unique_config_by_readdir(
    discovery: &ConfigDiscovery,
    dir: &Path,
) -> Result<Option<DiscoveredConfigFile>, ConfigConflict> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };

    // Cache the supported names once; the iteration body needs only name comparison.
    let config_names = discovery.config_file_names();
    let mut matches = Vec::new();

    for entry in entries.flatten() {
        let name = entry.file_name();
        if !config_names.iter().any(|n| name == *n) {
            continue;
        }

        let Ok(file_type) = entry.file_type() else { continue };
        // Intentional: skip directories, symlinks, sockets, ... — only
        // regular files are considered configs (matches walker's
        // `follow_links(false)`).
        #[expect(clippy::filetype_is_file)]
        if !file_type.is_file() {
            continue;
        }

        if let Some(config) = discovery.discover_config_file(&entry.path()) {
            matches.push(config);
        }
    }

    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.into_iter().next()),
        _ => Err(ConfigConflict::new(dir.to_path_buf(), matches)),
    }
}

// ---

// Shared on-demand nested-config detection infrastructure.
//
// State is centralized in `NestedConfigCtx` so Phase 2 (direct file targets),
// Phase 3 (parallel walk), and the stdin path all share one cache (each load
// runs at most once walk-wide) and one signal (`any_config_found`).

/// Result of loading a direct config in a single directory.
type ConfigLoadResult = Result<Option<Arc<ConfigResolver>>, String>;

/// Walk-wide shared cache for direct-config loads.
///
/// Each entry's `OnceLock` ensures the underlying load runs at most once per
/// directory across all visitors and across phases.
type ConfigLoadCache = Arc<Mutex<FxHashMap<PathBuf, Arc<OnceLock<ConfigLoadResult>>>>>;

/// Walk-wide shared map of "directory has a direct config" entries.
///
/// **Lock discipline**: never hold this lock across a `ConfigLoadCache` load.
/// Acquire the read/write lock, do the lookup or insert, release immediately.
type ScopeByDir = Arc<RwLock<FxHashMap<PathBuf, Arc<ConfigResolver>>>>;

/// Walk-wide shared cache for the parsed `.editorconfig`.
///
/// Loaded on first access (via `OnceLock`) and cloned per nested-config load
/// instead of re-reading and re-parsing the same file for every probed dir.
/// `Err` is cached too, so a malformed `.editorconfig` is not retried.
type EditorconfigCache = Arc<OnceLock<Result<Option<EditorConfig>, String>>>;

/// Shared nested-config detection context: cached `ConfigDiscovery` + load
/// cache + scope map + loader inputs (`editorconfig_path`, `js_config_loader`)
/// + the `any_config_found` signal.
///
/// Cloning is shallow (each field is already `Arc` / `Copy`).
#[derive(Clone)]
pub struct NestedConfigCtx {
    discovery: ConfigDiscovery,
    editorconfig_path: Option<Arc<Path>>,
    editorconfig_cache: EditorconfigCache,
    #[cfg(feature = "napi")]
    js_config_loader: Option<JsConfigLoaderCb>,
    any_config_found: Arc<AtomicBool>,
    scope_by_dir: ScopeByDir,
    config_load_cache: ConfigLoadCache,
}

impl NestedConfigCtx {
    pub fn new(
        editorconfig_path: Option<Arc<Path>>,
        #[cfg(feature = "napi")] js_config_loader: Option<JsConfigLoaderCb>,
    ) -> Self {
        Self {
            discovery: config_discovery(),
            editorconfig_path,
            editorconfig_cache: Arc::new(OnceLock::new()),
            #[cfg(feature = "napi")]
            js_config_loader,
            any_config_found: Arc::new(AtomicBool::new(false)),
            scope_by_dir: Arc::new(RwLock::new(FxHashMap::default())),
            config_load_cache: Arc::new(Mutex::new(FxHashMap::default())),
        }
    }

    /// Get the parsed `.editorconfig`, loading once and reusing the cached
    /// `EditorConfig` (or cached `Err`) for every subsequent caller.
    fn cached_editorconfig(&self) -> Result<Option<EditorConfig>, String> {
        self.editorconfig_cache
            .get_or_init(|| load_editorconfig(self.editorconfig_path.as_deref()))
            .clone()
    }

    /// Returns `true` if `path`'s file name matches a supported config file.
    pub fn is_config_file(&self, path: &Path) -> bool {
        self.discovery.discover_config_file(path).is_some()
    }

    /// Look up a registered scope for `dir` without probing.
    pub fn lookup_scope(&self, dir: &Path) -> Option<Arc<ConfigResolver>> {
        self.scope_by_dir.read().expect("scope_by_dir rwlock poisoned").get(dir).cloned()
    }

    pub fn config_found(&self) -> bool {
        self.any_config_found.load(Ordering::Relaxed)
    }

    /// Read `scope_by_dir` for `dir`; on miss, probe via the load cache and
    /// register the result.
    ///
    /// Returns:
    /// - `Ok(Some(_))` — `dir` has a direct config (registered).
    /// - `Ok(None)` — `dir` has no direct config, or Vite+ `.fmt` missing.
    /// - `Err(_)` — load / parse / validate failure.
    ///
    /// On a successful load, the `any_config_found` signal is set and the
    /// resolver is inserted into `scope_by_dir` (only the first writer wins).
    pub fn probe_dir(&self, dir: &Path) -> Result<Option<Arc<ConfigResolver>>, String> {
        if let Some(hit) = self.lookup_scope(dir) {
            return Ok(Some(hit));
        }

        match self.get_or_load_direct_config(dir)? {
            Some(loaded) => {
                self.any_config_found.store(true, Ordering::Relaxed);
                let mut guard = self.scope_by_dir.write().expect("scope_by_dir rwlock poisoned");
                guard.entry(dir.to_path_buf()).or_insert_with(|| Arc::clone(&loaded));
                Ok(Some(loaded))
            }
            None => Ok(None),
        }
    }

    /// Get-or-compute the direct-config load result for `dir`, dedupe walk-wide.
    ///
    /// `OnceLock::get_or_init` blocks concurrent callers for the same `dir`
    /// until the first init completes. `Ok(Some(_))` / `Ok(None)` / `Err(_)`
    /// are all cached, so broken configs are not retried and "no config in
    /// this dir" lookups stay O(1).
    fn get_or_load_direct_config(&self, dir: &Path) -> ConfigLoadResult {
        // Acquire (or insert) the cell, then drop the outer mutex immediately.
        let cell = {
            let mut guard =
                self.config_load_cache.lock().expect("config_load_cache mutex poisoned");
            let entry = guard.entry(dir.to_path_buf()).or_insert_with(|| Arc::new(OnceLock::new()));
            Arc::clone(entry)
        };

        cell.get_or_init(|| {
            let editorconfig = self.cached_editorconfig()?;
            load_direct_config_in_dir(
                &self.discovery,
                dir,
                editorconfig,
                #[cfg(feature = "napi")]
                self.js_config_loader.as_ref(),
            )
        })
        .clone()
    }
}

/// Load and validate a config file located **directly** inside `dir`.
///
/// Returns `Ok(None)` when `dir` has no supported config file, or when a
/// `vite.config.ts` is present but lacks a `.fmt` field (Vite+ mode). This
/// matches `discover_config`'s "skip and continue" semantics applied at a
/// single-dir scope.
fn load_direct_config_in_dir(
    discovery: &ConfigDiscovery,
    dir: &Path,
    editorconfig: Option<EditorConfig>,
    #[cfg(feature = "napi")] js_config_loader: Option<&JsConfigLoaderCb>,
) -> Result<Option<Arc<ConfigResolver>>, String> {
    let Some(config_file) = find_unique_config_by_readdir(discovery, dir)
        .map_err(|e| Into::<oxc_diagnostics::OxcDiagnostic>::into(e).to_string())?
    else {
        return Ok(None);
    };

    let load_err = |err: String| format!("Failed to load config in {}: {err}", dir.display());

    let Some(mut resolver) = build_resolver_from_discovered(
        config_file,
        editorconfig,
        #[cfg(feature = "napi")]
        js_config_loader,
    )
    .map_err(load_err)?
    else {
        return Ok(None);
    };

    resolver.build_and_validate().map_err(load_err)?;
    Ok(Some(Arc::new(resolver)))
}
