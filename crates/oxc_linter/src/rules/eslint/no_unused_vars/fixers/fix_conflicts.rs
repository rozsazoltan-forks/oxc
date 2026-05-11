use lazy_regex::Regex;
use oxc_str::{CompactStr, Ident};

use super::{NoUnusedVars, Symbol};
use crate::rules::eslint::no_unused_vars::options::IgnorePattern;

impl NoUnusedVars {
    pub(super) fn get_unused_arg_name(&self, symbol: &Symbol<'_, '_>) -> Option<CompactStr> {
        Self::get_unused_name(symbol, &self.args_ignore_pattern)
    }

    pub(super) fn get_unused_var_name(&self, symbol: &Symbol<'_, '_>) -> Option<CompactStr> {
        Self::get_unused_name(symbol, &self.vars_ignore_pattern)
    }

    /// Build a replacement name that satisfies the configured ignore pattern
    /// and does not collide with another binding in the same scope.
    ///
    /// This currently supports the default ignore pattern and explicit `^_`
    /// patterns:
    ///
    /// ```js
    /// function foo(unused, _unused) {}
    /// // `unused` becomes `_unused0`, not `_unused`.
    /// ```
    ///
    /// More complex ignore patterns are intentionally skipped until the fixer
    /// can reliably synthesize a matching identifier.
    fn get_unused_name(
        symbol: &Symbol<'_, '_>,
        ignore_pattern: &IgnorePattern<Regex>,
    ) -> Option<CompactStr> {
        let ignored_name: String = match ignore_pattern.as_ref() {
            // TODO: support more patterns
            IgnorePattern::Default => {
                format!("_{}", symbol.name())
            }
            IgnorePattern::Some(re) if re.as_str() == "^_" => {
                format!("_{}", symbol.name())
            }
            _ => return None,
        };

        // adjust name to avoid conflicts
        let scopes = symbol.scoping();
        let scope_id = symbol.scope_id();
        let mut i = 0;
        let mut new_name = ignored_name.clone();
        while scopes.scope_has_binding(scope_id, Ident::from(new_name.as_str())) {
            new_name = format!("{ignored_name}{i}");
            i += 1;
        }

        Some(new_name.into())
    }
}
