use std::{collections::HashMap, ops::{BitAnd, Shl, Shr}, thread};
use std::error::Error;
use std::ops::{BitOr, BitXor, Deref, Not};
use std::sync::{mpsc, Arc};
use std::time::Instant;
use crate::field::{FieldOperations, FieldOps, M};
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField};
use rand::Rng;
use ruint::aliases::U256;
use serde::{Deserialize, Serialize};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, Validate};
use compiler::intermediate_representation::ir_interface::OperatorType;
use ruint::uint;

fn ark_se<S, A: CanonicalSerialize>(a: &A, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut bytes = vec![];
    a.serialize_with_mode(&mut bytes, Compress::Yes)
        .map_err(serde::ser::Error::custom)?;
    s.serialize_bytes(&bytes)
}

fn ark_de<'de, D, A: CanonicalDeserialize>(data: D) -> Result<A, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let s: Vec<u8> = serde::de::Deserialize::deserialize(data)?;
    let a = A::deserialize_with_mode(s.as_slice(), Compress::Yes, Validate::Yes);
    a.map_err(serde::de::Error::custom)
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Operation {
    Mul,
    Div,
    Add,
    Sub,
    Pow,
    Idiv,
    Mod,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    Land,
    Lor,
    Shl,
    Shr,
    Bor,
    Band,
    Bxor,
}

impl Operation {
    pub fn eval(&self, a: U256, b: U256) -> U256 {
        use Operation::*;
        match self {
            Mul => a.mul_mod(b, M),
            Div => {
                if b == U256::ZERO {
                    // as we are simulating a circuit execution with signals
                    // values all equal to 0, just return 0 here in case of
                    // division by zero
                    U256::ZERO
                } else {
                    a.mul_mod(b.inv_mod(M).unwrap(), M)
                }
            },
            Add => a.add_mod(b, M),
            Sub => a.add_mod(M - b, M),
            Pow => a.pow_mod(b, M),
            Mod => a.div_rem(b).1,
            Eq => U256::from(a == b),
            Neq => U256::from(a != b),
            Lt => u_lt(&a, &b),
            Gt => u_gt(&a, &b),
            Leq => u_lte(&a, &b),
            Geq => u_gte(&a, &b),
            Land => U256::from(a != U256::ZERO && b != U256::ZERO),
            Lor => U256::from(a != U256::ZERO || b != U256::ZERO),
            Shl => compute_shl_uint(a, b),
            Shr => compute_shr_uint(a, b),
            // TODO test with conner case when it is possible to get the number
            //      bigger then modulus
            Bor => a.bitor(b),
            Band => a.bitand(b),
            // TODO test with conner case when it is possible to get the number
            //      bigger then modulus
            Bxor => a.bitxor(b),
            Idiv => if b == U256::ZERO { U256::ZERO } else { a / b },
        }
    }
}

impl From<&Operation> for crate::proto::DuoOp {
    fn from(v: &Operation) -> Self {
        match v {
            Operation::Mul => crate::proto::DuoOp::Mul,
            Operation::Div => crate::proto::DuoOp::Div,
            Operation::Add => crate::proto::DuoOp::Add,
            Operation::Sub => crate::proto::DuoOp::Sub,
            Operation::Pow => crate::proto::DuoOp::Pow,
            Operation::Idiv => crate::proto::DuoOp::Idiv,
            Operation::Mod => crate::proto::DuoOp::Mod,
            Operation::Eq => crate::proto::DuoOp::Eq,
            Operation::Neq => crate::proto::DuoOp::Neq,
            Operation::Lt => crate::proto::DuoOp::Lt,
            Operation::Gt => crate::proto::DuoOp::Gt,
            Operation::Leq => crate::proto::DuoOp::Leq,
            Operation::Geq => crate::proto::DuoOp::Geq,
            Operation::Land => crate::proto::DuoOp::Land,
            Operation::Lor => crate::proto::DuoOp::Lor,
            Operation::Shl => crate::proto::DuoOp::Shl,
            Operation::Shr => crate::proto::DuoOp::Shr,
            Operation::Bor => crate::proto::DuoOp::Bor,
            Operation::Band => crate::proto::DuoOp::Band,
            Operation::Bxor => crate::proto::DuoOp::Bxor,
        }
    }
}

impl TryFrom<OperatorType> for Operation {
    type Error = String;
    fn try_from(op: OperatorType) -> Result<Self, Self::Error> {
        match op {
            OperatorType::Mul => Ok(Operation::Mul),
            OperatorType::Div => Ok(Operation::Div),
            OperatorType::Add => Ok(Operation::Add),
            OperatorType::Sub => Ok(Operation::Sub),
            OperatorType::Pow => Ok(Operation::Pow),
            OperatorType::IntDiv => Ok(Operation::Idiv),
            OperatorType::Mod => Ok(Operation::Mod),
            OperatorType::ShiftL => Ok(Operation::Shl),
            OperatorType::ShiftR => Ok(Operation::Shr),
            OperatorType::LesserEq => Ok(Operation::Leq),
            OperatorType::GreaterEq => Ok(Operation::Geq),
            OperatorType::Lesser => Ok(Operation::Lt),
            OperatorType::Greater => Ok(Operation::Gt),
            OperatorType::Eq(1) => Ok(Operation::Eq),
            OperatorType::Eq(_) => todo!(),
            OperatorType::NotEq => Ok(Operation::Neq),
            OperatorType::BoolOr => Ok(Operation::Lor),
            OperatorType::BoolAnd => Ok(Operation::Land),
            OperatorType::BitOr => Ok(Operation::Bor),
            OperatorType::BitAnd => Ok(Operation::Band),
            OperatorType::BitXor => Ok(Operation::Bxor),
            OperatorType::PrefixSub => Err("Not a binary operation".to_string()),
            OperatorType::BoolNot => Err("Not a binary operation".to_string()),
            OperatorType::Complement => Err("Not a binary operation".to_string()),
            OperatorType::ToAddress => Err("Not a binary operation".to_string()),
            OperatorType::MulAddress => Ok(Operation::Mul),
            OperatorType::AddAddress => Ok(Operation::Add),
        }
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UnoOperation {
    Neg,
    Id, // identity - just return self
    Lnot,
    Bnot,
}

impl UnoOperation {
    pub fn eval(&self, a: U256) -> U256 {
        match self {
            UnoOperation::Neg => if a == U256::ZERO { U256::ZERO } else { M - a },
            UnoOperation::Id => a,
            UnoOperation::Lnot => if a == U256::ZERO {
                uint!(1_U256)
            } else {
                U256::ZERO
            },
            UnoOperation::Bnot => {
                let a = a.not();
                let mask = U256::ZERO.not().shr(M.leading_zeros());
                let a = a & mask;
                if a >= M { a - M } else { a }
            },
        }
    }
}

impl From<&UnoOperation> for crate::proto::UnoOp {
    fn from(v: &UnoOperation) -> Self {
        match v {
            UnoOperation::Neg => crate::proto::UnoOp::Neg,
            UnoOperation::Id => crate::proto::UnoOp::Id,
            UnoOperation::Lnot => crate::proto::UnoOp::Lnot,
            UnoOperation::Bnot => crate::proto::UnoOp::Bnot,
        }
    }
}

impl TryFrom<OperatorType> for UnoOperation {
    type Error = String;
    fn try_from(op: OperatorType) -> Result<Self, Self::Error> {
        let err = Err("Not an unary operation".to_string());
        match op {
            OperatorType::Mul => err,
            OperatorType::Div => err,
            OperatorType::Add => err,
            OperatorType::Sub => err,
            OperatorType::Pow => err,
            OperatorType::IntDiv => err,
            OperatorType::Mod => err,
            OperatorType::ShiftL => err,
            OperatorType::ShiftR => err,
            OperatorType::LesserEq => err,
            OperatorType::GreaterEq => err,
            OperatorType::Lesser => err,
            OperatorType::Greater => err,
            OperatorType::Eq(1) => err,
            OperatorType::Eq(_) => err,
            OperatorType::NotEq => err,
            OperatorType::BoolOr => err,
            OperatorType::BoolAnd => err,
            OperatorType::BitOr => err,
            OperatorType::BitAnd => err,
            OperatorType::BitXor => err,
            OperatorType::PrefixSub => Ok(UnoOperation::Neg),
            OperatorType::BoolNot => Ok(UnoOperation::Lnot),
            OperatorType::Complement => Ok(UnoOperation::Bnot),
            OperatorType::ToAddress => Ok(UnoOperation::Id),
            OperatorType::MulAddress => err,
            OperatorType::AddAddress => err,
        }
    }
}


#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TresOperation {
    TernCond,
}

impl TresOperation {
    pub fn eval(&self, a: U256, b: U256, c: U256) -> U256 {
        match self {
            TresOperation::TernCond => if a == U256::ZERO { c } else { b },
        }
    }
}

impl From<&TresOperation> for crate::proto::TresOp {
    fn from(v: &TresOperation) -> Self {
        match v {
            TresOperation::TernCond => crate::proto::TresOp::TernCond,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Node {
    #[default]
    Unknown,
    Input(usize),
    Constant(U256),
    Constant2(usize),
    #[serde(serialize_with = "ark_se", deserialize_with = "ark_de")]
    MontConstant(Fr),
    UnoOp(UnoOperation, usize),
    Op(Operation, usize, usize),
    TresOp(TresOperation, usize, usize, usize),
}

// TODO remove pub from Vec<Node>
#[derive(Default)]
pub struct Nodes(pub Vec<Node>);

impl Nodes {
    pub fn new() -> Self {
        Nodes(Vec::new())
    }

    pub fn to_const(&self, idx: NodeIdx) -> Result<U256, NodeConstErr> {
        let me = self.0.get(idx.0).ok_or(NodeConstErr::EmptyNode(idx))?;
        match me {
            Node::Unknown => panic!("Unknown node"),
            Node::Constant(v) => Ok(*v),
            Node::Constant2(_) => todo!(),
            Node::UnoOp(op, a) => {
                Ok(op.eval(
                    self.to_const(NodeIdx(*a))?))
            }
            Node::Op(op, a, b) => {
                Ok(op.eval(
                    self.to_const(NodeIdx(*a))?,
                    self.to_const(NodeIdx(*b))?))
            }
            Node::TresOp(op, a, b, c) => {
                Ok(op.eval(
                    self.to_const(NodeIdx(*a))?,
                    self.to_const(NodeIdx(*b))?,
                    self.to_const(NodeIdx(*c))?))
            }
            Node::Input(_) => Err(NodeConstErr::InputSignal),
            Node::MontConstant(_) => {
                panic!("MontConstant should not be used here")
            }
        }
    }

    pub fn push(&mut self, n: Node) -> NodeIdx {
        self.0.push(n);
        NodeIdx(self.0.len() - 1)
    }

    pub fn get(&self, idx: NodeIdx) -> Option<&Node> {
        self.0.get(idx.0)
    }
}

impl Deref for Nodes {
    type Target = Vec<Node>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Copy, Clone)]
pub struct NodeIdx(pub usize);

impl From<usize> for NodeIdx {
    fn from(v: usize) -> Self {
        NodeIdx(v)
    }
}

#[derive(Debug)]
pub enum NodeConstErr {
    EmptyNode(NodeIdx),
    InputSignal,
}

impl std::fmt::Display for NodeConstErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeConstErr::EmptyNode(idx) => {
                write!(f, "empty node at index {}", idx.0)
            }
            NodeConstErr::InputSignal => {
                write!(f, "input signal is not a constant")
            }
        }
    }
}

impl Error for NodeConstErr {}

fn compute_shl_uint(a: U256, b: U256) -> U256 {
    debug_assert!(b.lt(&U256::from(256)));
    let ls_limb = b.as_limbs()[0];
    a.shl(ls_limb as usize)
}

fn compute_shr_uint(a: U256, b: U256) -> U256 {
    debug_assert!(b.lt(&U256::from(256)));
    let ls_limb = b.as_limbs()[0];
    a.shr(ls_limb as usize)
}

/// All references must be backwards.
fn assert_valid(nodes: &[Node]) {
    for (i, &node) in nodes.iter().enumerate() {
        if let Node::Op(_, a, b) = node {
            assert!(a < i);
            assert!(b < i);
        } else if let Node::UnoOp(_, a) = node {
            assert!(a < i);
        } else if let Node::TresOp(_, a, b, c) = node {
            assert!(a < i);
            assert!(b < i);
            assert!(c < i);
        }
    }
}

pub fn optimize(nodes: &mut Vec<Node>, outputs: &mut [usize]) {
    tree_shake(nodes, outputs);
    propagate(nodes);
    value_numbering(nodes, outputs);
    constants(nodes);
    tree_shake(nodes, outputs);
}

pub fn evaluate(nodes: &[Node], inputs: &[U256], outputs: &[usize]) -> Vec<U256> {
    // assert_valid(nodes);

    let start = Instant::now();
    // Evaluate the graph.
    let mut values = Vec::with_capacity(nodes.len());
    for &node in nodes.iter() {
        let value = match node {
            Node::Unknown => panic!("Unknown node"),
            Node::Constant(c) => c,
            Node::Constant2(_) => todo!(),
            Node::MontConstant(c) => fr_to_u256(&c),
            Node::Input(i) => inputs[i],
            Node::Op(op, a, b) => op.eval(values[a], values[b]),
            Node::UnoOp(op, a) => op.eval(values[a]),
            Node::TresOp(op, a, b, c) => op.eval(values[a], values[b], values[c]),
        };
        values.push(value);
    }

    let r = outputs.iter().map(|&i| values[i]).collect();
    println!("graph calculated in {:?}", start.elapsed());
    r
}

pub fn evaluate2<T: FieldOps, F: FieldOperations<Type = T>>(
    ff: F, nodes: &[Node], inputs: &[T], outputs: &[usize],
    constants: &[T]) -> Vec<T>
where Vec<T>: FromIterator<<F as FieldOperations>::Type>
{
    // assert_valid(nodes);

    let start = Instant::now();
    // Evaluate the graph.
    let mut values = Vec::with_capacity(nodes.len());
    for &node in nodes.iter() {
        let value = match node {
            Node::Unknown => panic!("Unknown node"),
            Node::Constant(_) => todo!("remove embedded constants"),
            Node::Constant2(i) => constants[i],
            Node::MontConstant(_) => todo!("remove usage of montgomery constants"),
            Node::Input(i) => inputs[i],
            Node::Op(op, a, b) => {
                match op {
                    Operation::Mul => ff.mul(values[a], values[b]),
                    Operation::Div => ff.div(values[a], values[b]),
                    Operation::Add => ff.add(values[a], values[b]),
                    Operation::Sub => ff.sub(values[a], values[b]),
                    Operation::Pow => ff.pow(values[a], values[b]),
                    Operation::Idiv => ff.idiv(values[a], values[b]),
                    Operation::Mod => ff.modulo(values[a], values[b]),
                    Operation::Eq => ff.eq(values[a], values[b]),
                    Operation::Neq => ff.neq(values[a], values[b]),
                    Operation::Lt => ff.lt(values[a], values[b]),
                    Operation::Gt => ff.gt(values[a], values[b]),
                    Operation::Leq => ff.lte(values[a], values[b]),
                    Operation::Geq => ff.gte(values[a], values[b]),
                    Operation::Land => ff.land(values[a], values[b]),
                    Operation::Lor => ff.lor(values[a], values[b]),
                    Operation::Shl => ff.shl(values[a], values[b]),
                    Operation::Shr => ff.shr(values[a], values[b]),
                    Operation::Bor => ff.bor(values[a], values[b]),
                    Operation::Band => ff.band(values[a], values[b]),
                    Operation::Bxor => ff.bxor(values[a], values[b]),
                }
            },
            Node::UnoOp(op, a) => {
                match op {
                    UnoOperation::Neg => ff.neg(values[a]),
                    UnoOperation::Id => values[a],
                    UnoOperation::Lnot => ff.lnot(values[a]),
                    UnoOperation::Bnot => ff.bnot(values[a]),
                }
            },
            Node::TresOp(op, a, b, c) => {
                match op {
                    TresOperation::TernCond => {
                        if values[a].is_zero() { values[c] } else { values[b] }
                    },
                }
            },
        };
        values.push(value);
    }

    let r = outputs.iter().map(|&i| values[i]).collect();
    println!("generic typed graph calculated in {:?}", start.elapsed());
    r
}

pub fn evaluate_parallel(nodes: &[Node], inputs: &[U256], outputs: &[usize]) -> Vec<U256> {
    let start = Instant::now();
    let inputs: Arc<[U256]> = Arc::from(inputs);
    println!("total nodes: {}", nodes.len());
    let mut nodes_splitted = 0;
    let sz = outputs.len() / 4;

    let mut outputs2 = Vec::new();
    let mut subgraphs = Vec::new();

    for (i, chunk) in outputs.chunks(sz).enumerate() {
        let mut nodes = Vec::from(nodes);
        let mut chunk = Vec::from(chunk);
        tree_shake(&mut nodes, &mut chunk);
        nodes_splitted += nodes.len();
        println!("chunk #{}: {} nodes", i, nodes.len());

        outputs2.push(chunk);
        subgraphs.push(nodes);
    }
    println!("total nodes splitted: {}", nodes_splitted);
    println!("graph splitted in {:?}", start.elapsed());
    // assert_valid(nodes);

    let start = Instant::now();

    let mut handles = Vec::new();
    let threads_num = subgraphs.len();
    let (tx, rx) = mpsc::channel();
    for (i, (nodes, outputs)) in subgraphs.into_iter().zip(outputs2).enumerate() {
        let inputs = Arc::clone(&inputs);
        let tx = tx.clone();
        let handle = thread::spawn(move || {
            let mut values = Vec::with_capacity(nodes.len());
            for &node in nodes.iter() {
                let value = match node {
                    Node::Unknown => panic!("Unknown node"),
                    Node::Constant(c) => c,
                    Node::Constant2(_) => todo!(),
                    Node::MontConstant(_) => todo!("remove montgomery constants"),
                    Node::Input(i) => inputs[i],
                    Node::Op(op, a, b) => op.eval(values[a], values[b]),
                    Node::UnoOp(op, a) => op.eval(values[a]),
                    Node::TresOp(op, a, b, c) => op.eval(values[a], values[b], values[c]),
                };
                values.push(value);
            }

            let witness_signals: Vec<U256> = outputs.iter().map(|&i| values[i]).collect();
            tx.send((i, witness_signals)).unwrap();
        });
        handles.push(handle);
    }

    let mut final_results = vec![Vec::new(); threads_num];

    for handle in handles {
        handle.join().unwrap();
    }

    for _ in 0..threads_num {
        if let Ok((i, signals)) = rx.recv() {
            final_results[i] = signals;
        }
    }

    let r = final_results.into_iter().flatten().collect();
    println!("graph calculated in parallel in {:?}", start.elapsed());

    r
}

/// Constant propagation
pub fn propagate(nodes: &mut [Node]) {
    assert_valid(nodes);
    let mut constants = 0_usize;
    for i in 0..nodes.len() {
        if let Node::Op(op, a, b) = nodes[i] {
            if let (Node::Constant(va), Node::Constant(vb)) = (nodes[a], nodes[b]) {
                nodes[i] = Node::Constant(op.eval(va, vb));
                constants += 1;
            } else if a == b {
                // Not constant but equal
                use Operation::*;
                if let Some(c) = match op {
                    Eq | Leq | Geq => Some(true),
                    Neq | Lt | Gt => Some(false),
                    _ => None,
                } {
                    nodes[i] = Node::Constant(U256::from(c));
                    constants += 1;
                }
            }
        } else if let Node::UnoOp(op, a) = nodes[i] {
            if let Node::Constant(va) = nodes[a] {
                nodes[i] = Node::Constant(op.eval(va));
                constants += 1;
            }
        } else if let Node::TresOp(op, a, b, c) = nodes[i] {
            if let (Node::Constant(va), Node::Constant(vb), Node::Constant(vc)) = (nodes[a], nodes[b], nodes[c]) {
                nodes[i] = Node::Constant(op.eval(va, vb, vc));
                constants += 1;
            }
        }
    }

    eprintln!("Propagated {constants} constants");
}

/// Remove unused nodes
pub fn tree_shake(nodes: &mut Vec<Node>, outputs: &mut [usize]) {
    assert_valid(nodes);

    // Mark all nodes that are used.
    let mut used = vec![false; nodes.len()];
    for &i in outputs.iter() {
        used[i] = true;
    }

    // Work backwards from end as all references are backwards.
    for i in (0..nodes.len()).rev() {
        if used[i] {
            if let Node::Op(_, a, b) = nodes[i] {
                used[a] = true;
                used[b] = true;
            }
            if let Node::UnoOp(_, a) = nodes[i] {
                used[a] = true;
            }
            if let Node::TresOp(_, a, b, c) = nodes[i] {
                used[a] = true;
                used[b] = true;
                used[c] = true;
            }
        }
    }

    // Remove unused nodes
    let n = nodes.len();
    let mut retain = used.iter();
    nodes.retain(|_| *retain.next().unwrap());
    let removed = n - nodes.len();

    // Renumber references.
    let mut renumber = vec![None; n];
    let mut index = 0;
    for (i, &used) in used.iter().enumerate() {
        if used {
            renumber[i] = Some(index);
            index += 1;
        }
    }
    assert_eq!(index, nodes.len());
    for (&used, renumber) in used.iter().zip(renumber.iter()) {
        assert_eq!(used, renumber.is_some());
    }

    // Renumber references.
    for node in nodes.iter_mut() {
        if let Node::Op(_, a, b) = node {
            *a = renumber[*a].unwrap();
            *b = renumber[*b].unwrap();
        }
        if let Node::UnoOp(_, a) = node {
            *a = renumber[*a].unwrap();
        }
        if let Node::TresOp(_, a, b, c) = node {
            *a = renumber[*a].unwrap();
            *b = renumber[*b].unwrap();
            *c = renumber[*c].unwrap();
        }
    }
    for output in outputs.iter_mut() {
        *output = renumber[*output].unwrap();
    }

    eprintln!("Removed {removed} unused nodes");
}

/// Randomly evaluate the graph
fn random_eval(nodes: &mut [Node]) -> Vec<U256> {
    let mut rng = rand::thread_rng();
    let mut values = Vec::with_capacity(nodes.len());
    let mut inputs = HashMap::new();
    let mut prfs = HashMap::new();
    let mut prfs_uno = HashMap::new();
    let mut prfs_tres = HashMap::new();
    for node in nodes.iter() {
        use Operation::*;
        let value = match node {
            Node::Unknown => panic!("Unknown node"),

            // Constants evaluate to themselves
            Node::Constant(c) => *c,

            Node::Constant2(_) => todo!(),

            Node::MontConstant(_) => unimplemented!("should not be used"),

            // Algebraic Ops are evaluated directly
            // Since the field is large, by Swartz-Zippel if
            // two values are the same then they are likely algebraically equal.
            Node::Op(op @ (Add | Sub | Mul), a, b) => op.eval(values[*a], values[*b]),

            // Input and non-algebraic ops are random functions
            // TODO: https://github.com/recmo/uint/issues/95 and use .gen_range(..M)
            Node::Input(i) => *inputs.entry(*i).or_insert_with(|| rng.gen::<U256>() % M),
            Node::Op(op, a, b) => *prfs
                .entry((*op, values[*a], values[*b]))
                .or_insert_with(|| rng.gen::<U256>() % M),
            Node::UnoOp(op, a) => *prfs_uno
                .entry((*op, values[*a]))
                .or_insert_with(|| rng.gen::<U256>() % M),
            Node::TresOp(op, a, b, c) => *prfs_tres
                .entry((*op, values[*a], values[*b], values[*c]))
                .or_insert_with(|| rng.gen::<U256>() % M),
        };
        values.push(value);
    }
    values
}

/// Value numbering
pub fn value_numbering(nodes: &mut [Node], outputs: &mut [usize]) {
    assert_valid(nodes);

    // Evaluate the graph in random field elements.
    let values = random_eval(nodes);

    // Find all nodes with the same value.
    let mut value_map = HashMap::new();
    for (i, &value) in values.iter().enumerate() {
        value_map.entry(value).or_insert_with(Vec::new).push(i);
    }

    // For nodes that are the same, pick the first index.
    let mut renumber = Vec::with_capacity(nodes.len());
    for value in values {
        renumber.push(value_map[&value][0]);
    }

    // Renumber references.
    for node in nodes.iter_mut() {
        if let Node::Op(_, a, b) = node {
            *a = renumber[*a];
            *b = renumber[*b];
        }
        if let Node::UnoOp(_, a) = node {
            *a = renumber[*a];
        }
        if let Node::TresOp(_, a, b, c) = node {
            *a = renumber[*a];
            *b = renumber[*b];
            *c = renumber[*c];
        }
    }
    for output in outputs.iter_mut() {
        *output = renumber[*output];
    }

    eprintln!("Global value numbering applied");
}

/// Probabilistic constant determination
pub fn constants(nodes: &mut [Node]) {
    assert_valid(nodes);

    // Evaluate the graph in random field elements.
    let values_a = random_eval(nodes);
    let values_b = random_eval(nodes);

    // Find all nodes with the same value.
    let mut constants = 0;
    for i in 0..nodes.len() {
        if let Node::Constant(_) = nodes[i] {
            continue;
        }
        if values_a[i] == values_b[i] {
            nodes[i] = Node::Constant(values_a[i]);
            constants += 1;
        }
    }
    eprintln!("Found {} constants", constants);
}

// M / 2
const halfM: U256 = uint!(10944121435919637611123202872628637544274182200208017171849102093287904247808_U256);


fn u_gte(a: &U256, b: &U256) -> U256 {
    let a_neg = &halfM < a;
    let b_neg = &halfM < b;

    match (a_neg, b_neg) {
        (false, false) => U256::from(a >= b),
        (true, false) => uint!(0_U256),
        (false, true) => uint!(1_U256),
        (true, true) => U256::from(a >= b),
    }
}

fn u_lte(a: &U256, b: &U256) -> U256 {
    let a_neg = &halfM < a;
    let b_neg = &halfM < b;

    match (a_neg, b_neg) {
        (false, false) => U256::from(a <= b),
        (true, false) => uint!(1_U256),
        (false, true) => uint!(0_U256),
        (true, true) => U256::from(a <= b),
    }
}

fn u_gt(a: &U256, b: &U256) -> U256 {
    let a_neg = &halfM < a;
    let b_neg = &halfM < b;

    match (a_neg, b_neg) {
        (false, false) => U256::from(a > b),
        (true, false) => uint!(0_U256),
        (false, true) => uint!(1_U256),
        (true, true) => U256::from(a > b),
    }
}

fn u_lt(a: &U256, b: &U256) -> U256 {
    let a_neg = &halfM < a;
    let b_neg = &halfM < b;

    match (a_neg, b_neg) {
        (false, false) => U256::from(a < b),
        (true, false) => uint!(1_U256),
        (false, true) => uint!(0_U256),
        (true, true) => U256::from(a < b),
    }
}

pub fn fr_to_u256(x: &Fr) -> U256 {
    U256::from_limbs(x.into_bigint().0)
}

pub fn u256_to_fr(x: &U256) -> Fr {
    Fr::from_bigint(BigInt::new(x.into_limbs())).unwrap()
}

#[cfg(test)]
mod tests {
    use std::ops::{Div};
    use super::*;
    use ruint::{uint};

    #[test]
    fn test_div() {
        assert_eq!(
            Operation::Div.eval(U256::from(2u64), U256::from(3u64)),
            U256::from_str_radix("7296080957279758407415468581752425029516121466805344781232734728858602831873", 10).unwrap());

        assert_eq!(
            Operation::Div.eval(U256::from(6u64), U256::from(2u64)),
            U256::from_str_radix("3", 10).unwrap());

        assert_eq!(
            Operation::Div.eval(U256::from(7u64), U256::from(2u64)),
            U256::from_str_radix("10944121435919637611123202872628637544274182200208017171849102093287904247812", 10).unwrap());
    }

    #[test]
    fn test_idiv() {
        assert_eq!(
            Operation::Idiv.eval(U256::from(2u64), U256::from(3u64)),
            U256::from(0));

        assert_eq!(
            Operation::Idiv.eval(U256::from(6u64), U256::from(2u64)),
            U256::from(3));

        assert_eq!(
            Operation::Idiv.eval(U256::from(7u64), U256::from(2u64)),
            U256::from(3));
    }

    #[test]
    fn test_fr_mod() {
        assert_eq!(
            Operation::Mod.eval(U256::from(7u64), U256::from(2u64)),
            U256::from(1));

        assert_eq!(
            Operation::Mod.eval(U256::from(7u64), U256::from(9u64)),
            U256::from(7));
    }

    #[test]
    fn test_greater_then_module() {
        // println!("{}", Fr::MODULUS);
        // let f = Fr::from_str("21888242871839275222246405745257275088548364400416034343698204186575808495619").unwrap();
        // println!("[2] {}", f);
        // let mut i = f.into_bigint();
        // println!("[3] {}", i);
        // let j = i.add_with_carry(&Fr::MODULUS);
        // println!("[4] {}", i);
        // println!("[5] {}", j);
        // if i.cmp(&Fr::MODULUS).is_ge() {
        //     i.sub_with_borrow(&Fr::MODULUS);
        // }
        // let f2 = Fr::from_bigint(i).unwrap();
        // println!("[6] {}", f2);
        // let a= Fr::from(4u64);
        // let b= Fr::from(2u64);
        // let c = shl(a, b);
        // assert_eq!(c.cmp(&Fr::from(16u64)), Ordering::Equal)
    }

    #[test]
    fn test_u_gte() {
        let result = u_gte(&uint!(10_U256), &uint!(3_U256));
        assert_eq!(result, uint!(1_U256));

        let result = u_gte(&uint!(3_U256), &uint!(3_U256));
        assert_eq!(result, uint!(1_U256));

        let result = u_gte(&uint!(2_U256), &uint!(3_U256));
        assert_eq!(result, uint!(0_U256));

        // -1 >= 3 => 0
        let result = u_gte(
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495616_U256),
            &uint!(3_U256));
        assert_eq!(result, uint!(0_U256));

        // -1 >= -2 => 1
        let result = u_gte(
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495616_U256),
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495615_U256));
        assert_eq!(result, uint!(1_U256));

        // -2 >= -1 => 0
        let result = u_gte(
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495615_U256),
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495616_U256));
        assert_eq!(result, uint!(0_U256));

        // -2 == -2 => 1
        let result = u_gte(
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495615_U256),
            &uint!(21888242871839275222246405745257275088548364400416034343698204186575808495615_U256));
        assert_eq!(result, uint!(1_U256));
    }

    #[test]
    fn test_x() {
        let x = M.div(uint!(2_U256));

        println!("x: {:?}", x.as_limbs());
        println!("x: {}", M);
    }

    #[test]
    fn test_2() {
        let nodes: Vec<Node> = vec![];
        // let node = nodes[0];
        let node = nodes.get(0);
        println!("{:?}", node);
    }

    #[test]
    fn test_pow() {
        let a = uint!(21888242871839275222246405745257275088548364400416034343698204186575808495615_U256);
        let b = uint!(21888_U256);
        let c = Operation::Pow.eval(a, b);
        let want = uint!(6741803673964058984617537840767809723100020752467791363717299927390655464193_U256);
        assert_eq!(c, want);
    }

    #[test]
    fn test_bnot() {
        assert_eq!(
            uint!(7059779437489773633646340506914701874769131765994106666166191815402473914366_U256),
            UnoOperation::Bnot.eval(uint!(0_U256)));
        assert_eq!(
            uint!(7059779437489773633646340506914701874769131765994106666166191815400326430719_U256),
            UnoOperation::Bnot.eval(uint!(2147483647_U256)));
        assert_eq!(
            uint!(7059779437489773633646340506914701874769131765994106666166191815402473914367_U256),
            UnoOperation::Bnot.eval(uint!(21888242871839275222246405745257275088548364400416034343698204186575808495616_U256)));
        assert_eq!(
            uint!(7059779437489773633646340506914701874769131765994106666166191815401042258601_U256),
            UnoOperation::Bnot.eval(uint!(1431655765_U256)));
        assert_eq!(
            uint!(7059779437489773633646340506914701874769131765994106666166191815404191901285_U256),
            UnoOperation::Bnot.eval(uint!(21888242871839275222246405745257275088548364400416034343698204186574090508698_U256)));
        assert_eq!(
            uint!(0_U256),
            UnoOperation::Bnot.eval(uint!(115792089237316195423570985008687907853269984665640564039457584007913129639935_U256)));
        assert_eq!(
            uint!(19298681539552699237261830834781317975544997444273427339909597334652188273322_U256),
            UnoOperation::Bnot.eval(uint!(38597363079105398474523661669562635951089994888546854679819194669304376546645_U256)));
        assert_eq!(
            uint!(17368813385597429313535647751303186177990497699846084605918637601186969445990_U256),
            UnoOperation::Bnot.eval(uint!(69475253542389717254142591005212744711961990799384338423674550404747877783961_U256)));
        assert_eq!(
            uint!(16975279050329094783283862284904804026119806273934822715754654203603313563979_U256),
            UnoOperation::Bnot.eval(uint!(11972743258999954072608883967267172937197689892475318294109741798374968846004_U256)));
        assert_eq!(
            uint!(10364945975102880683525514432911402591886023268641012016029012611469420464237_U256),
            UnoOperation::Bnot.eval(uint!(18583076334226168172367231819260574371431472897769128993835383390508861945746_U256)));
        assert_eq!(
            uint!(4253782056457656234530291275605853130160190710592122558439987573692654305887_U256),
            UnoOperation::Bnot.eval(uint!(2805997381032117399116049231308848744608941055401984107726204241709819608479_U256)));
    }

    #[test]
    fn test_lnot() {
        assert_eq!(
            uint!(0_U256),
            UnoOperation::Lnot.eval(uint!(1_U256)));
        assert_eq!(
            uint!(1_U256),
            UnoOperation::Lnot.eval(uint!(0_U256)));
        assert_eq!(
            uint!(0_U256),
            UnoOperation::Lnot.eval(uint!(10944121435919637611123202872628637544274182200208017171849102093287904247808_U256)));
        assert_eq!(
            uint!(0_U256),
            UnoOperation::Lnot.eval(uint!(115792089237316195423570985008687907853269984665640564039457584007913129639935_U256)));
    }

    #[test]
    fn test_half() {
        // let h = M.div(U256::from(2));
        let h = M.wrapping_shr(1);
        type BN254 = ruint::Uint<254, 4>;

        let m = BN254::from_str_radix(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10).unwrap();
        let a = BN254::from_str_radix(
            "18583076334226168172367231819260574371431472897769128993835383390508861945746",
            10).unwrap();
        let a = a.not();
        // let mask = BN254::ZERO.not().shr(m.leading_zeros());
        // let a = a & mask;
        let a = if a >= m { a - m } else { a };

        let want = BN254::from_str_radix(
            "10364945975102880683525514432911402591886023268641012016029012611469420464237",
            10
        ).unwrap();
        assert_eq!(want, a);

        assert_eq!(h, halfM);
    }
}
