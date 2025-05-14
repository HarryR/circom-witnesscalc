use std::collections::HashMap;
use std::error::Error;
use crate::field::{FieldOperations, FieldOps};

#[repr(u8)]
#[derive(Debug)]
pub enum OpCode {
    NoOp = 0,
    // Put signals to the stack
    // required stack_i64: signal index
    LoadSignal      = 1,
    // Store the signal
    // stack_ff contains the value to store
    // stack_i64 contains the signal index
    StoreSignal     = 2,
    PushI64         = 3, // Push i64 value to the stack
    PushFf          = 4, // Push ff value to the stack
    // Set variables from the stack
    // arguments:      offset from the base pointer
    // required stack: values to store equal to variables number from arguments
    StoreVariableFf = 5,
    LoadVariableI64 = 6,
    LoadVariableFf  = 7,
    OpMul           = 8,
    OpAdd           = 9,
    OpNeq           = 10,
}

pub struct Circuit<T: FieldOps> {
    pub main_template_id: usize,
    pub templates: Vec<Template>,
    pub prime: T,
    pub witness: Vec<usize>,
    pub input_signals_info: HashMap<String, usize>,
    pub signals_num: usize,
}

pub struct Template {
    pub name: String,
    pub code: Vec<u8>,
    pub vars_i64_num: usize,
    pub vars_ff_num: usize,
}

fn read_instruction(code: &[u8], ip: usize) -> OpCode {
    unsafe { std::mem::transmute::<u8, OpCode>(code[ip]) }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("Stack is empty")]
    StackUnderflow,
    #[error("Stack is not large enough")]
    StackOverflow,
    #[error("Value on the stack is None")]
    StackVariableIsNotSet,
    #[error("Failed to convert from i32 to usize")]
    I32ToUsizeConversion,
    #[error("Signal index is out of bounds")]
    SignalIndexOutOfBounds,
    #[error("Signal is not set")]
    SignalIsNotSet,
    #[error("Signal is already set")]
    SignalIsAlreadySet,
    #[error("Code index is out of bounds")]
    CodeIndexOutOfBounds,
}

struct VM<T: FieldOps> {
    stack_ff: Vec<Option<T>>,
    stack_i64: Vec<Option<i64>>,
    base_pointer_ff: usize,
    // base_pointer_i64: usize,
}

impl<T: FieldOps> VM<T> {
    fn new() -> Self {
        Self {
            stack_ff: Vec::new(),
            stack_i64: Vec::new(),
            base_pointer_ff: 0,
            // base_pointer_i64: 0,
        }
    }

    fn push_ff(&mut self, value: T) {
        self.stack_ff.push(Some(value));
    }

    fn pop_ff(&mut self) -> Result<T, RuntimeError> {
        self.stack_ff.pop().ok_or(RuntimeError::StackUnderflow)?
            .ok_or(RuntimeError::StackVariableIsNotSet)
    }

    fn push_i64(&mut self, value: i64) {
        self.stack_i64.push(Some(value));
    }

    fn pop_i64(&mut self) -> Result<i64, RuntimeError> {
        self.stack_i64
            .pop().ok_or(RuntimeError::StackUnderflow)?
            .ok_or(RuntimeError::StackVariableIsNotSet)
    }

    fn pop_usize(&mut self) -> Result<usize, RuntimeError> {
        self.pop_i64()?
            .try_into()
            .map_err(|_| RuntimeError::I32ToUsizeConversion)
    }
}

// Converts 8 bytes from the code to i64 and then to usize. Returns error
// if the code length is too short or if i64 < 0 or if i64 is too big to fit
// into usize.
fn usize_from_code(
    code: &[u8], ip: usize) -> Result<(usize, usize), RuntimeError> {

    let slice = code.get(ip..ip+size_of::<u64>())
        .ok_or(RuntimeError::CodeIndexOutOfBounds)?;
    let bytes: [u8; 8] = slice.try_into()
        .map_err(|_| RuntimeError::I32ToUsizeConversion)?;
    let v = i64::from_le_bytes(bytes);
    let v: usize = v.try_into()
        .map_err(|_| RuntimeError::I32ToUsizeConversion)?;

    Ok((v, ip+8))
}

pub fn disassemble_instruction<T>(
    code: &[u8], ip: usize, name: &str) -> usize
where
    T: FieldOps {

    let op_code = read_instruction(code, ip);
    let mut ip = ip + 1usize;

    print!("{:08x} [{:10}] ", ip, name);

    match op_code {
        OpCode::NoOp => {
            println!("NoOp");
        }
        OpCode::LoadSignal => {
            println!("LoadSignal");
        }
        OpCode::StoreSignal => {
            println!("StoreSignal");
        }
        OpCode::PushI64 => {
            let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
            ip += size_of::<i64>();
            println!("PushI64: {}", v);
        }
        OpCode::PushFf => {
            let s = &code[ip..ip+T::BYTES];
            ip += T::BYTES;
            let v = T::from_le_bytes(s).unwrap();
            println!("PushFf: {}", v);
        }
        OpCode::StoreVariableFf => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            println!("StoreVariableFf: {}", var_idx);
        }
        OpCode::LoadVariableI64 => {
            println!("LoadVariableI64");
            todo!();
        }
        OpCode::LoadVariableFf => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            println!("LoadVariableFf: {}", var_idx);
        }
        OpCode::OpMul => {
            println!("OpMul");
        }
        OpCode::OpAdd => {
            println!("OpAdd");
        }
        OpCode::OpNeq => {
            println!("OpNeq");
        }
    }

    ip
}

pub fn execute<F: FieldOperations>(
    templates: &[Template], signals: &mut [Option<F::Type>],
    main_template_id: usize, ff: F) -> Result<(), Box<dyn Error>> {

    let mut ip: usize = 0;
    let mut vm = VM::<F::Type>::new();
    vm.stack_ff.resize_with(
        templates[main_template_id].vars_ff_num, || None);
    vm.stack_i64.resize_with(
        templates[main_template_id].vars_i64_num, || None);
    let template_signals_start = 1usize;

    'label: loop {
        if ip == templates[main_template_id].code.len() {
            break 'label;
        }

        disassemble_instruction::<F::Type>(
            &templates[main_template_id].code, ip,
            &templates[main_template_id].name);

        let op_code = read_instruction(
            &templates[main_template_id].code, ip);
        ip += 1;

        match op_code {
            OpCode::NoOp => (),
            OpCode::LoadSignal => {
                let signal_idx = template_signals_start + vm.pop_usize()?;
                let s = signals.get(signal_idx)
                    .ok_or(RuntimeError::SignalIndexOutOfBounds)?
                    .ok_or(RuntimeError::SignalIsNotSet)?;

                vm.push_ff(s);
            }
            OpCode::StoreSignal => {
                let signal_idx = template_signals_start + vm.pop_usize()?;
                if signal_idx >= signals.len() {
                    return Err(Box::new(RuntimeError::SignalIndexOutOfBounds));
                }
                if signals[signal_idx].is_some() {
                    return Err(Box::new(RuntimeError::SignalIsAlreadySet));
                }
                signals[signal_idx] = Some(vm.pop_ff()?);
            }
            OpCode::PushI64 => {
                vm.push_i64(
                    i64::from_le_bytes(
                        (&templates[main_template_id].code[ip..ip+8])
                            .try_into().unwrap()));
                ip += 8;
            }
            OpCode::PushFf => {
                let s = &templates[main_template_id]
                    .code[ip..ip+F::Type::BYTES];
                ip += F::Type::BYTES;
                let v = ff.parse_le_bytes(s)?;
                vm.push_ff(v);
            }
            OpCode::StoreVariableFf => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[main_template_id].code, ip)?;
                let value = vm.pop_ff()?;
                vm.stack_ff[vm.base_pointer_ff + var_idx] = Some(value);
            }
            OpCode::LoadVariableI64 => {
                todo!();
            }
            OpCode::LoadVariableFf => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[main_template_id].code, ip)?;
                let var = match vm.stack_ff.get(vm.base_pointer_ff + var_idx) {
                    Some(v) => v,
                    None => return Err(Box::new(RuntimeError::StackOverflow)),
                };
                let var = match var {
                    Some(v) => v,
                    None => return Err(Box::new(RuntimeError::StackVariableIsNotSet)),
                };
                vm.push_ff(*var);
            }
            OpCode::OpMul => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.mul(lhs, rhs));
            }
            OpCode::OpAdd => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.add(lhs, rhs));
            }
            OpCode::OpNeq => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.neq(lhs, rhs));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_ok() {
        let x: &[u8] = &[1, 2, 3];
        let x2 = x.get(2..4);

        println!("{:?}", x2);
    }
}