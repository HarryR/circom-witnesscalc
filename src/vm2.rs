use std::collections::HashMap;
use std::error::Error;
use crate::field::{Field, FieldOperations, FieldOps};

#[repr(u8)]
#[derive(Debug)]
pub enum OpCode {
    NoOp                 = 0,
    // Put signals to the stack
    // required stack_i64: signal index
    LoadSignal           = 1,
    // Store the signal
    // stack_ff contains the value to store
    // stack_i64 contains the signal index
    StoreSignal          = 2,
    PushI64              = 3, // Push i64 value to the stack
    PushFf               = 4, // Push ff value to the stack
    // Set variables from the stack
    // arguments:      offset from the base pointer
    // required stack: values to store equal to variables number from arguments
    StoreVariableFf      = 5,
    LoadVariableI64      = 6,
    LoadVariableFf       = 7,
    // Jump to the instruction
    // arguments:      4 byte LE offset to jump
    // required stack: the ff value to check for failure
    JumpIfFalse          = 8,
    // Jump to the instruction
    // arguments:      4 byte LE offset to jump
    Jump                 = 9,
    // stack_i64 contains the error code
    Error                = 10,
    // Get the component signal and put it to the stack_ff
    // stack_i64:0 contains the signal index
    // stack_i64:-1 contains the component index
    LoadCmpSignal        = 11,
    // Store the component signal and run
    // stack_ff contains the value to store
    // stack_i64:0 contains the signal index
    // stack_i64:-1 contains the component index
    StoreCmpSignalAndRun = 12,
    OpMul                = 13,
    OpAdd                = 14,
    OpNeq                = 15,
    OpDiv                = 16,
    OpSub                = 17,
    OpEq                 = 18,
    OpEqz                = 19,
}

pub struct Component {
    pub signals_start: usize,
    pub template_id: usize,
    pub components: Vec<Option<Box<Component>>>,
    pub number_of_inputs: usize,
}

pub struct Circuit<T: FieldOps> {
    pub main_template_id: usize,
    pub templates: Vec<Template>,
    pub field: Field<T>,
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
    #[error("component is not initialized")]
    UninitializedComponent,
    #[error("assertion: {0}")]
    Assertion(i64),
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
        OpCode::JumpIfFalse => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            println!("JumpIfFalse: {}", v);
        }
        OpCode::Jump => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            println!("Jump: {}", v);
        }
        OpCode::LoadCmpSignal => {
            println!("LoadCmpSignal");
        }
        OpCode::StoreCmpSignalAndRun => {
            println!("StoreCmpSignalAndRun");
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
        OpCode::OpDiv => {
            println!("OpDiv");
        }
        OpCode::OpSub => {
            println!("OpSub");
        }
        OpCode::OpEq => {
            println!("OpEq");
        }
        OpCode::OpEqz => {
            println!("OpEqz");
        }
        OpCode::Error => {
            println!("Error");
        }
    }

    ip
}

pub fn execute<F, T: FieldOps>(
    templates: &[Template], signals: &mut [Option<T>], ff: &F,
    component_tree: &mut Component) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations<Type = T> {

    let mut ip: usize = 0;
    let mut vm = VM::<T>::new();
    vm.stack_ff.resize_with(
        templates[component_tree.template_id].vars_ff_num, || None);
    vm.stack_i64.resize_with(
        templates[component_tree.template_id].vars_i64_num, || None);

    'label: loop {
        if ip == templates[component_tree.template_id].code.len() {
            break 'label;
        }

        disassemble_instruction::<T>(
            &templates[component_tree.template_id].code, ip,
            &templates[component_tree.template_id].name);

        let op_code = read_instruction(
            &templates[component_tree.template_id].code, ip);
        ip += 1;

        match op_code {
            OpCode::NoOp => (),
            OpCode::LoadSignal => {
                let signal_idx = component_tree.signals_start + vm.pop_usize()?;
                let s = signals.get(signal_idx)
                    .ok_or(RuntimeError::SignalIndexOutOfBounds)?
                    .ok_or(RuntimeError::SignalIsNotSet)?;

                vm.push_ff(s);
            }
            OpCode::StoreSignal => {
                let signal_idx = component_tree.signals_start + vm.pop_usize()?;
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
                        (&templates[component_tree.template_id].code[ip..ip+8])
                            .try_into().unwrap()));
                ip += 8;
            }
            OpCode::PushFf => {
                let s = &templates[component_tree.template_id]
                    .code[ip..ip+T::BYTES];
                ip += T::BYTES;
                let v = ff.parse_le_bytes(s)?;
                vm.push_ff(v);
            }
            OpCode::StoreVariableFf => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[component_tree.template_id].code, ip)?;
                let value = vm.pop_ff()?;
                vm.stack_ff[vm.base_pointer_ff + var_idx] = Some(value);
            }
            OpCode::LoadVariableI64 => {
                todo!();
            }
            OpCode::LoadVariableFf => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[component_tree.template_id].code, ip)?;
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
            OpCode::LoadCmpSignal => {
                let sig_idx = vm.pop_usize()?;
                let cmp_idx = vm.pop_usize()?;
                vm.push_ff(match component_tree.components[cmp_idx] {
                    None => {
                        return Err(
                            Box::new(RuntimeError::UninitializedComponent))
                    }
                    Some(ref c) => {
                        signals[c.signals_start + sig_idx].ok_or(RuntimeError::SignalIsNotSet)?
                    }
                });
            }
            OpCode::StoreCmpSignalAndRun => {
                let sig_idx = vm.pop_usize()?;
                let cmp_idx = vm.pop_usize()?;
                let value = vm.pop_ff()?;
                match component_tree.components[cmp_idx] {
                    None => {
                        return Err(
                            Box::new(RuntimeError::UninitializedComponent))
                    }
                    Some(ref mut c) => {
                        signals[c.signals_start + sig_idx] = Some(value);
                        c.number_of_inputs -= 1;
                        execute(templates, signals, ff, c)?;
                    }
                }
            }
            OpCode::JumpIfFalse => {
                let offset_bytes = &templates[component_tree.template_id].code[ip..ip + size_of::<i32>()];
                let offset = i32::from_le_bytes((offset_bytes).try_into().unwrap());
                ip += size_of::<i32>();

                if vm.pop_ff()?.is_zero() {
                    if offset < 0 {
                        ip -= offset.unsigned_abs() as usize;
                    } else {
                        ip += offset as usize;
                    }
                }
            }
            OpCode::Error => {
                let error_code = vm.pop_i64()?;
                return Err(Box::new(RuntimeError::Assertion(error_code)));
            }
            OpCode::Jump => {
                let offset_bytes = &templates[component_tree.template_id].code[ip..ip + size_of::<i32>()];
                let offset = i32::from_le_bytes((offset_bytes).try_into().unwrap());
                ip += size_of::<i32>();

                if offset < 0 {
                    ip -= offset.unsigned_abs() as usize;
                } else {
                    ip += offset as usize;
                }
            }
            OpCode::OpDiv => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.div(lhs, rhs));
            }
            OpCode::OpSub => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.sub(lhs, rhs));
            }
            OpCode::OpEq => {
                let lhs = vm.pop_ff()?;
                let rhs = vm.pop_ff()?;
                vm.push_ff(ff.eq(lhs, rhs));
            }
            OpCode::OpEqz => {
                let arg = vm.pop_ff()?;
                if arg.is_zero() {
                    vm.push_ff(T::one());
                } else {
                    vm.push_ff(T::zero());
                }
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