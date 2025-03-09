use std::io::{Cursor, Error, ErrorKind};
use ruint::aliases::U256;
use crate::graph::{Node, Operation, TresOperation, UnoOperation};
use crate::InputSignalsInfo;
use crate::storage::{read_message, WriteBackReader, WITNESSCALC_GRAPH_MAGIC};

// deserialize_witnesscalc_graph_from_bytes is almost the same as
// deserialize_witnesscalc_graph but with custom implemented protobuf parser
// specifically optimized to unpack the list of Nodes.
pub fn deserialize_witnesscalc_graph_from_bytes(
    bytes: &[u8]
) -> std::io::Result<(Vec<Node>, Vec<usize>, InputSignalsInfo)> {

    if bytes.len() < WITNESSCALC_GRAPH_MAGIC.len() {
        return Err(Error::new(ErrorKind::Other, "Invalid magic"));
    }
    if !bytes[..WITNESSCALC_GRAPH_MAGIC.len()].eq(WITNESSCALC_GRAPH_MAGIC) {
        return Err(Error::new(ErrorKind::Other, "Invalid magic"));
    }

    let mut idx: usize = WITNESSCALC_GRAPH_MAGIC.len();
    let nodes_num = u64::from_le_bytes(bytes[idx..idx+8].try_into().unwrap());
    idx += 8;

    let mut nodes = Vec::with_capacity(nodes_num as usize);
    for _ in 0..nodes_num {
        let (msg_len, int_len) = decode_varint_u32(&bytes[idx..])?;
        idx += int_len;
        nodes.push(decode_node(&bytes[idx..idx+msg_len as usize])?);
        idx += msg_len as usize;
    }

    let r = Cursor::new(&bytes[idx..]);
    let mut br = WriteBackReader::new(r);
    let md: crate::proto::GraphMetadata = read_message(&mut br)?;

    let witness_signals = md.witness_signals
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();

    let input_signals = md.inputs.iter()
        .map(|(k, v)| {
            (k.clone(), (v.offset as usize, v.len as usize))
        })
        .collect::<InputSignalsInfo>();

    Ok((nodes, witness_signals, input_signals))
}

#[repr(u8)]
#[derive(Debug)]
enum WireType {
    VARINT = 0,
    I64 = 1,
    LEN = 2,
    SGROUP = 3,
    EGROUP = 4,
    I32 = 5,
}

impl TryFrom<u8> for WireType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(WireType::VARINT),
            1 => Ok(WireType::I64),
            2 => Ok(WireType::LEN),
            3 => Ok(WireType::SGROUP),
            4 => Ok(WireType::EGROUP),
            5 => Ok(WireType::I32),
            _ => Err(()),
        }
    }
}

/// Decodes a protobuf Node message into a Node enum
pub fn decode_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Empty input buffer",
        ));
    }

    let (field_number, wire_type, tag_size) = read_tag(bytes)?;

    if !matches!(wire_type, WireType::LEN) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!(
                "Expected length-delimited field: field_number={}, wire_type={:?}",
                field_number, wire_type),
        ));
    }
    let bytes = &bytes[tag_size..];

    let (length, varint_size) = decode_varint_u32(bytes)?;
    let bytes = &bytes[varint_size..];
    if bytes.len() != length as usize {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incorrect ConstantNode field size",
        ));
    }

    match field_number {
        1 => decode_input_node(bytes),
        2 => decode_constant_node(bytes),
        3 => decode_uno_op_node(bytes),
        4 => decode_duo_op_node(bytes),
        5 => decode_tres_op_node(bytes),
        _ => {
            panic!("found unknown node")
        }
    }
}

fn decode_input_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Ok(Node::Input(0));
    }

    let (field_number, wire_type, tag_size) = read_tag(bytes)?;

    if field_number != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected field number 1 for InputNode",
        ));
    }

    if !matches!(wire_type, WireType::VARINT) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected length-delimited field for InputNode",
        ));
    }

    let bytes = &bytes[tag_size..];

    let (value, varint_size) = decode_varint_u32(bytes)?;
    if varint_size != bytes.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Incorrect InputNode field size",
        ));
    }
    Ok(Node::Input(value as usize))
}

fn decode_big_uint(bytes: &[u8]) -> Result<U256, Error> {
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Empty input buffer",
        ));
    }

    let (field_number, wire_type, tag_size) = read_tag(bytes)?;
    if field_number != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected field number 1 for BigUInt",
        ));
    }
    if !matches!(wire_type, WireType::LEN) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected length-delimited field for BigUInt",
        ));
    }

    let bytes = &bytes[tag_size..];

    let (length, varint_size) = decode_varint_u32(bytes)?;
    if bytes.len() - varint_size != length as usize {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incorrect BigUInt field size",
        ));
    }
    let bytes = &bytes[varint_size..];
    Ok(U256::from_le_slice(bytes))
}

/// Decodes a UnoOpNode message into an Operation and two indices
fn decode_uno_op_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Ok(Node::UnoOp(UnoOperation::Neg, 0));
    }

    let mut offset = 0;

    let mut op = UnoOperation::Neg;
    let mut a_idx: usize = 0;

    // Process all fields in the message
    while offset < bytes.len() {
        let (field_number, wire_type, tag_size) = read_tag(&bytes[offset..])?;
        offset += tag_size;

        if !matches!(wire_type, WireType::VARINT) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Expected varint as DuoOpNode field",
            ));
        }

        let (value, varint_size) = decode_varint_u32(&bytes[offset..])?;
        offset += varint_size;

        match field_number {
            1 => {
                op = match value {
                    0 => UnoOperation::Neg,
                    1 => UnoOperation::Id,
                    _ => return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Unknown DuoOp operation value: {}", value),
                    )),
                };
            },
            2 => {
                a_idx = value as usize;
            },
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Unknown UnoOpNode tag"));
            }
        }
    }

    Ok(Node::UnoOp(op, a_idx))
}

/// Decodes a DuoOpNode message into an Operation and two indices
fn decode_duo_op_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Ok(Node::Op(Operation::Mul, 0, 0));
    }

    let mut offset = 0;

    let mut op = Operation::Mul;
    let mut a_idx: usize = 0;
    let mut b_idx: usize = 0;

    // Process all fields in the message
    while offset < bytes.len() {
        let (field_number, wire_type, tag_size) = read_tag(&bytes[offset..])?;
        offset += tag_size;

        if !matches!(wire_type, WireType::VARINT) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Expected varint as DuoOpNode field",
            ));
        }

        let (value, varint_size) = decode_varint_u32(&bytes[offset..])?;
        offset += varint_size;

        match field_number {
            1 => {
                op = match value {
                    0 => Operation::Mul,
                    1 => Operation::Div,
                    2 => Operation::Add,
                    3 => Operation::Sub,
                    4 => Operation::Pow,
                    5 => Operation::Idiv,
                    6 => Operation::Mod,
                    7 => Operation::Eq,
                    8 => Operation::Neq,
                    9 => Operation::Lt,
                    10 => Operation::Gt,
                    11 => Operation::Leq,
                    12 => Operation::Geq,
                    13 => Operation::Land,
                    14 => Operation::Lor,
                    15 => Operation::Shl,
                    16 => Operation::Shr,
                    17 => Operation::Bor,
                    18 => Operation::Band,
                    19 => Operation::Bxor,
                    _ => return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Unknown DuoOp operation value: {}", value),
                    )),
                };
            },
            2 => {
                a_idx = value as usize;
            },
            3 => {
                b_idx = value as usize;
            },
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Unknown DuoOpNode tag"));
            }
        }
    }

    Ok(Node::Op(op, a_idx, b_idx))
}

fn decode_tres_op_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Ok(Node::TresOp(TresOperation::TernCond, 0, 0, 0));
    }

    let mut offset = 0;

    let mut op = TresOperation::TernCond;
    let mut a_idx: usize = 0;
    let mut b_idx: usize = 0;
    let mut c_idx: usize = 0;

    // Process all fields in the message
    while offset < bytes.len() {
        let (field_number, wire_type, tag_size) = read_tag(&bytes[offset..])?;
        offset += tag_size;

        if !matches!(wire_type, WireType::VARINT) {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Expected varint as TresOpNode field",
            ));
        }

        let (value, varint_size) = decode_varint_u32(&bytes[offset..])?;
        offset += varint_size;

        match field_number {
            1 => {
                op = match value {
                    0 => TresOperation::TernCond,
                    _ => return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("Unknown TresOp operation value: {}", value),
                    )),
                };
            },
            2 => {
                a_idx = value as usize;
            },
            3 => {
                b_idx = value as usize;
            },
            4 => {
                c_idx = value as usize;
            },
            _ => {
                return Err(Error::new(ErrorKind::InvalidData, "Unknown TresOpNode tag"));
            }
        }
    }

    Ok(Node::TresOp(op, a_idx, b_idx, c_idx))
}

fn decode_constant_node(bytes: &[u8]) -> Result<Node, Error> {
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Empty input buffer",
        ));
    }

    let (field_number, wire_type, tag_size) = read_tag(bytes)?;

    if field_number != 1 {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected field number 1 for ConstantNode",
        ));
    }

    if !matches!(wire_type, WireType::LEN) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Expected length-delimited field for ConstantNode",
        ));
    }

    let bytes = &bytes[tag_size..];
    let (length, varint_size) = decode_varint_u32(bytes)?;
    let bytes = &bytes[varint_size..];
    if bytes.len() != length as usize {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incorrect ConstantNode field size",
        ));
    }

    let n = decode_big_uint(bytes)?;

    Ok(Node::Constant(n))
}

fn read_tag(bytes: &[u8]) -> Result<(u32, WireType, usize), Error> {
    let (tag, consumed) = decode_varint_u32(bytes)?;
    let field_number = tag >> 3;
    let wire_type = TryFrom::<u8>::try_from((tag & 0x7) as u8).unwrap();
    Ok((field_number, wire_type, consumed))
}


#[inline]
pub fn decode_varint_u32(bytes: &[u8]) -> Result<(u32, usize), Error> {
    // Fast-path optimization for empty slices
    if bytes.is_empty() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Empty input buffer",
        ));
    }

    // Fast-path for single-byte varints (very common case)
    let first_byte = bytes[0];
    if first_byte < 0x80 {
        return Ok((first_byte as u32, 1));
    }

    // We need at least 2 bytes now
    if bytes.len() < 2 {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incomplete varint in the input",
        ));
    }

    // Unrolled loop for the remaining bytes - faster than iterating
    let mut result: u32 = (first_byte & 0x7F) as u32;

    let second_byte = bytes[1];
    if second_byte < 0x80 {
        result |= (second_byte as u32) << 7;
        return Ok((result, 2));
    }

    if bytes.len() < 3 {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incomplete varint in the input",
        ));
    }

    result |= ((second_byte & 0x7F) as u32) << 7;

    let third_byte = bytes[2];
    if third_byte < 0x80 {
        result |= (third_byte as u32) << 14;
        return Ok((result, 3));
    }

    if bytes.len() < 4 {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incomplete varint in the input",
        ));
    }

    result |= ((third_byte & 0x7F) as u32) << 14;

    let fourth_byte = bytes[3];
    if fourth_byte < 0x80 {
        result |= (fourth_byte as u32) << 21;
        return Ok((result, 4));
    }

    if bytes.len() < 5 {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            "Incomplete varint in the input",
        ));
    }

    result |= ((fourth_byte & 0x7F) as u32) << 21;

    let fifth_byte = bytes[4];
    // For u32, the fifth byte can only use 4 bits (plus the continuation bit)
    if fifth_byte > 0x0F {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Varint value exceeds u32::MAX"),
        ));
    }

    if fifth_byte < 0x80 {
        result |= (fifth_byte as u32) << 28;
        return Ok((result, 5));
    }

    // If we get here, the varint is invalid (too many continuation bits)
    Err(Error::new(
        ErrorKind::InvalidData,
        "Varint is too long for u32",
    ))
}
