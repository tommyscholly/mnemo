use std::collections::{HashMap, HashSet};

use graphviz_rust::{
    dot_generator::*,
    dot_structures::*,
    printer::{DotPrinter, PrinterContext},
};

use super::visit::visit_block_succs;
use crate::mir::{
    domtree::{DomTreeImpl, Searchable},
    *,
};

#[derive(Debug, Default)]
pub struct FlowNode {
    predecessors: HashSet<BlockId>,
    successors: HashSet<BlockId>,
}

pub struct FlowGraph {
    blocks: HashMap<BlockId, FlowNode>,
    entry: Option<BlockId>,
    dominators: HashMap<BlockId, Option<BlockId>>,
}

impl FlowGraph {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            entry: None,
            dominators: HashMap::new(),
        }
    }

    pub fn compute(&mut self, func: &Function) {
        self.blocks.clear();
        self.entry = None;

        if !func.blocks.is_empty() {
            self.entry = Some(func.blocks[0].block_id);
        }

        for block in func.blocks.iter() {
            self.blocks.entry(block.block_id).or_default();
        }

        for block in func.blocks.iter() {
            visit_block_succs(func, block, |to| {
                self.add_edge(block.block_id, to);
            });
        }

        if let Some(first_block) = func.blocks.first() {
            self.entry = Some(first_block.block_id);
        }
    }

    fn add_edge(&mut self, from: BlockId, to: BlockId) {
        self.blocks.entry(from).or_default().successors.insert(to);
        self.blocks.entry(to).or_default().predecessors.insert(from);
    }
}

impl Searchable<BlockId> for FlowGraph {
    type NodeId = BlockId;

    fn dfs<F>(&self, start: Self::NodeId, visit: &mut F)
    where
        F: FnMut(Self::NodeId),
    {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node) = stack.pop() {
            if !visited.contains(&node) {
                visited.insert(node);
                visit(node);
                stack.extend(self.blocks[&node].successors.iter());
            }
        }
    }

    fn nodes(&self) -> Vec<Self::NodeId> {
        self.blocks.keys().cloned().collect()
    }
}

impl DomTreeImpl<BlockId> for FlowGraph {
    fn dom(&self, start: Self::NodeId) -> Option<Self::NodeId> {
        *self.dominators.get(&start).unwrap_or(&None)
    }

    fn set_dom(&mut self, node: Self::NodeId, dom: Option<Self::NodeId>) {
        self.dominators.insert(node, dom);
    }

    fn reset_dom(&mut self) {
        self.dominators.iter_mut().for_each(|(_, dom)| *dom = None);
    }

    fn preds(&self, node: Self::NodeId) -> Vec<Self::NodeId> {
        self.blocks
            .get(&node)
            .unwrap()
            .predecessors
            .iter()
            .copied()
            .collect()
    }
}

// Graphviz functions
impl FlowGraph {
    pub fn to_dot(&self) -> Graph {
        let mut stmts: Vec<Stmt> = vec![
            stmt!(attr!("rankdir", "TB")),
            stmt!(attr!("fontname", esc "Helvetica")),
        ];

        if let Some(entry) = self.entry {
            stmts.push(stmt!(
                node!(esc format!("bb{}", entry); attr!("style", "filled"), attr!("fillcolor", "lightgreen"))
            ));
        }

        for block_id in self.blocks.keys() {
            let label = format!("BB{}", block_id);
            stmts.push(stmt!(
                node!(esc format!("bb{}", block_id); attr!("label", esc label))
            ));
        }

        for (from, flow_node) in &self.blocks {
            for to in &flow_node.successors {
                stmts.push(stmt!(
                    edge!(node_id!(esc format!("bb{}", from)) => node_id!(esc format!("bb{}", to)))
                ));
            }
        }

        graph!(strict di id!("FlowGraph"), stmts)
    }

    pub fn to_dot_string(&self) -> String {
        self.to_dot().print(&mut PrinterContext::default())
    }
}
