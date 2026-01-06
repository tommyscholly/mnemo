// structure for the frontend to pass around to track data

use std::{collections::HashMap, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Symbol(pub u32);

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Symbol {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

impl From<u32> for Symbol {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

// technically this should be in parse.rs
#[derive(Debug, Default)]
pub struct Ctx {
    strings: Vec<String>,
    indices: HashMap<String, Symbol>,
    pub parsing_region: bool,
}

impl Ctx {
    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.indices.get(s) {
            return sym;
        }

        let id = Symbol(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.indices.insert(s.to_string(), id);
        id
    }

    pub fn resolve(&self, sym: Symbol) -> &str {
        &self.strings[sym.0 as usize]
    }

    pub fn update(&mut self, sym: Symbol, s: &str) {
        self.strings[sym.0 as usize] = s.to_string();
    }
}
