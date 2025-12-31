// structure for the frontend to pass around to track data

use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Symbol(pub i32);

impl Symbol {
    pub fn new(id: i32) -> Self {
        Self(id)
    }
}

#[derive(Debug)]
pub struct Ctx {
    strings: Vec<String>,
    indices: HashMap<String, Symbol>,
    #[allow(unused)]
    reserved: HashMap<Symbol, String>,
}

impl Ctx {
    pub fn new() -> Self {
        let strings = vec![];
        let mut indices = HashMap::new();
        // reserved indices start at -1
        // indices.insert("println".to_string(), Symbol(-1));

        let mut reserved = HashMap::new();
        // reserved.insert(Symbol(-1), "println".to_string());
        Self {
            strings,
            indices,
            reserved,
        }
    }

    pub fn intern(&mut self, s: &str) -> Symbol {
        if let Some(&sym) = self.indices.get(s) {
            return sym;
        }

        let id = Symbol(self.strings.len() as i32);
        self.strings.push(s.to_string());
        self.indices.insert(s.to_string(), id);
        id
    }

    pub fn resolve(&self, sym: Symbol) -> &str {
        if sym.0 < 0 {
            return self.reserved.get(&sym).unwrap();
        }

        &self.strings[sym.0 as usize]
    }
}
