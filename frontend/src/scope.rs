use crate::ctx::Symbol;

pub type ScopeId = usize;
pub const ROOT_SCOPE: ScopeId = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Function(Symbol),
    Region(Option<Symbol>),
    Branch,
    Loop,
    Block,
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub id: ScopeId,
    pub parent: Option<ScopeId>,
    pub kind: ScopeKind,
}

impl Scope {
    pub fn new(id: ScopeId, parent: Option<ScopeId>, kind: ScopeKind) -> Self {
        Self { id, parent, kind }
    }
}

#[derive(Debug)]
pub struct ScopeTree {
    scopes: Vec<Scope>,
    current: ScopeId,
}

impl ScopeTree {
    pub fn new() -> Self {
        let root_scope = Scope::new(ROOT_SCOPE, None, ScopeKind::Block);
        Self {
            scopes: vec![root_scope],
            current: ROOT_SCOPE,
        }
    }

    pub fn enter(&mut self, kind: ScopeKind) -> ScopeId {
        let id = self.scopes.len();
        let scope = Scope::new(id, Some(self.current), kind);
        self.scopes.push(scope);
        self.current = id;
        id
    }

    pub fn exit(&mut self) {
        if let Some(parent) = self.scopes[self.current].parent {
            self.current = parent;
        }
    }

    pub fn current(&self) -> ScopeId {
        self.current
    }

    pub fn outlives(&self, a: ScopeId, b: ScopeId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        while let Some(parent) = self.scopes[current].parent {
            if parent == a {
                return true;
            }
            current = parent;
        }
        false
    }

    pub fn get_scope(&self, id: ScopeId) -> Option<&Scope> {
        self.scopes.get(id)
    }

    pub fn parent_of(&self, id: ScopeId) -> Option<ScopeId> {
        self.scopes.get(id).and_then(|s| s.parent)
    }
}

impl Default for ScopeTree {
    fn default() -> Self {
        Self::new()
    }
}
