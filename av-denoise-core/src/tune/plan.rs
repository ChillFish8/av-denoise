use std::collections::BTreeMap;

use super::key::SiteKey;

/// The candidate to run at each site key, by name.
///
/// A key with no entry runs its baseline. A site-wide entry applies to every key of that site
/// that has no entry of its own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelPlan {
    choices: BTreeMap<SiteKey, String>,
    site_choices: BTreeMap<String, String>,
}

impl KernelPlan {
    /// A plan that runs every site's baseline.
    pub fn baseline() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: SiteKey, name: &str) {
        let owned = name.to_string();
        self.choices.insert(key, owned);
    }

    /// Drops the entry of `key` itself, leaving any site-wide entry in place.
    pub fn remove(&mut self, key: &SiteKey) {
        self.choices.remove(key);
    }

    /// Runs `name` at every key of `site` that has no entry of its own.
    pub fn with_site(mut self, site: &str, name: &str) -> Self {
        let site_owned = site.to_string();
        let name_owned = name.to_string();
        self.site_choices.insert(site_owned, name_owned);
        self
    }

    pub fn get(&self, key: &SiteKey) -> Option<&str> {
        let exact = self.choices.get(key);
        let site = key.site();
        let site_wide = self.site_choices.get(site);

        exact.or(site_wide).map(String::as_str)
    }

    /// Copies in every entry of `other` this plan has none for.
    pub fn fill_from(&mut self, other: &KernelPlan) {
        for (key, name) in &other.choices {
            if !self.choices.contains_key(key) {
                let key_owned = key.clone();
                let name_owned = name.clone();
                self.choices.insert(key_owned, name_owned);
            }
        }

        for (site, name) in &other.site_choices {
            if !self.site_choices.contains_key(site) {
                let site_owned = site.clone();
                let name_owned = name.clone();
                self.site_choices.insert(site_owned, name_owned);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.choices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.choices.is_empty() && self.site_choices.is_empty()
    }
}

/// A named set of pipeline switches, compared by timing whole frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum PipelineProfile {
    /// The pipeline as it is built for the device.
    #[default]
    Baseline,
    /// The f16 search ring turned off on a device that supports it.
    F32Search,
    /// The collab search and filter as two kernels joined by a member buffer.
    SplitSearch,
}

impl PipelineProfile {
    pub const ALL: [PipelineProfile; 3] = [
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
        PipelineProfile::SplitSearch,
    ];

    /// The name the store writes.
    pub fn name(self) -> &'static str {
        match self {
            PipelineProfile::Baseline => "baseline",
            PipelineProfile::F32Search => "f32_search",
            PipelineProfile::SplitSearch => "split_search",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|profile| profile.name() == name)
    }
}
