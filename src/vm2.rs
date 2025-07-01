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
    // Set FF variable from the stack
    // arguments: offset from the base pointer
    // stack_ff:  value to store
    StoreVariableFf      = 5,
    LoadVariableFf       = 6,
    // Set I64 variable from the stack
    // arguments: offset from the base pointer
    // stack_i64:  value to store
    StoreVariableI64     = 7,
    LoadVariableI64      = 8,
    // Jump to the instruction if there is 0 on stack_ff
    // arguments: 4 byte LE offset to jump
    // stack_ff:  the value to check for failure
    JumpIfFalseFf        = 9,
    // Jump to the instruction if there is 0 on stack_i64
    // arguments: 4 byte LE offset to jump
    // stack_i64: the value to check for failure
    JumpIfFalseI64       = 10,
    // Jump to the instruction
    // arguments:      4 byte LE offset to jump
    Jump                 = 11,
    // stack_i64 contains the error code
    Error                = 12,
    // Get the component signal and put it to the stack_ff
    // stack_i64:0 contains the signal index
    // stack_i64:-1 contains the component index
    LoadCmpSignal        = 13,
    // Store the component signal and run
    // stack_ff contains the value to store
    // stack_i64:0 contains the signal index
    // stack_i64:-1 contains the component index
    StoreCmpSignalAndRun = 14,
    // Store the component input without decrementing input counter
    // stack_ff contains the value to store
    // stack_i64:0 contains the signal index
    // stack_i64:-1 contains the component index
    StoreCmpInput        = 15,
    OpMul                = 16,
    OpAdd                = 17,
    OpNeq                = 18,
    OpDiv                = 19,
    OpSub                = 20,
    OpEq                 = 21,
    OpEqz                = 22,
    OpI64Add             = 23,
    OpI64Sub             = 24,
    // Memory return operation
    // Copy data from source memory to destination memory
    // stack_i64:0 contains the size (number of elements)
    // stack_i64:-1 contains the source address
    // stack_i64:-2 contains the destination address
    FfMReturn            = 25,
    // Function call operation
    // arguments: 4-byte function index + 1-byte argument count
    // Then for each argument:
    //   1-byte argument type (0=i64 literal, 1=ff literal, 2=i64 memory, 3=ff memory)
    //   For literals: value bytes (8 for i64, T::BYTES for ff)
    //   For memory: 2 i64 addresses (addr and size)
    FfMCall              = 26,
    // Memory store operation (ff.store)
    // stack_ff:0 contains the value to store
    // stack_i64:0 contains the memory address
    FfStore              = 27,
    // Memory load operation (ff.load)
    // stack_i64:0 contains the memory address
    // Result pushed to stack_ff
    FfLoad               = 28,
    // Memory load operation (i64.load)
    // stack_i64:0 contains the memory address
    // Result pushed to stack_i64
    I64Load              = 29,
    // Field less-than comparison (ff.lt)
    // stack_ff:0 contains right operand
    // stack_ff:-1 contains left operand
    // Result pushed to stack_ff (1 if lhs < rhs, 0 otherwise)
    OpLt                 = 30,
    // Integer multiplication (i64.mul)
    // stack_i64:0 contains right operand
    // stack_i64:-1 contains left operand
    // Result pushed to stack_i64
    OpI64Mul             = 31,
    // Integer less-than-or-equal comparison (i64.le)
    // stack_i64:0 contains right operand
    // stack_i64:-1 contains left operand
    // Result pushed to stack_i64 (1 if lhs <= rhs, 0 otherwise)
    OpI64Lte             = 32,
    // Wrap field element to i64 (i64.wrap_ff)
    // stack_ff:0 contains the field element
    // Result pushed to stack_i64
    I64WrapFf            = 33,
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
    pub functions: Vec<Template>, // Functions are compiled the same way as templates
    pub function_registry: HashMap<String, usize>, // Function name -> index mapping
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
    base_pointer_i64: usize,
}

impl<T: FieldOps> VM<T> {
    fn new() -> Self {
        Self {
            stack_ff: Vec::new(),
            stack_i64: Vec::new(),
            base_pointer_ff: 0,
            base_pointer_i64: 0,
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
        OpCode::StoreVariableI64 => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            println!("StoreVariableI64: {}", var_idx);
        }
        OpCode::LoadVariableI64 => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            println!("LoadVariableI64: {}", var_idx);
        }
        OpCode::LoadVariableFf => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            println!("LoadVariableFf: {}", var_idx);
        }
        OpCode::JumpIfFalseFf => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            println!("JumpIfFalseFf: {}", v);
        }
        OpCode::JumpIfFalseI64 => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            println!("JumpIfFalseI64: {}", v);
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
        OpCode::StoreCmpInput => {
            println!("StoreCmpInput");
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
        OpCode::OpI64Add => {
            println!("OpI64Add");
        }
        OpCode::OpI64Sub => {
            println!("OpI64Sub");
        }
        OpCode::Error => {
            println!("Error");
        }
        OpCode::FfMReturn => {
            println!("FfMReturn");
        }
        OpCode::FfMCall => {
            // Read function index
            let func_idx = u32::from_le_bytes((&code[ip..ip+4]).try_into().unwrap());
            ip += 4;
            
            // Read argument count
            let arg_count = code[ip];
            ip += 1;
            
            print!("FfMCall: func_idx={}, args=[", func_idx);
            
            // Parse each argument
            for i in 0..arg_count {
                if i > 0 {
                    print!(", ");
                }
                
                let arg_type = code[ip];
                ip += 1;
                
                match arg_type {
                    0 => { // i64 literal
                        let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        print!("i64.{}", v);
                    }
                    1 => { // ff literal
                        let v = T::from_le_bytes(&code[ip..ip+T::BYTES]).unwrap();
                        ip += T::BYTES;
                        print!("ff.{}", v);
                    }
                    2 => { // i64 memory
                        let addr = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        let size = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        print!("i64.memory({},{})", addr, size);
                    }
                    3 => { // ff memory
                        let addr = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        let size = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        print!("ff.memory({},{})", addr, size);
                    }
                    _ => {
                        print!("unknown_arg_type({})", arg_type);
                    }
                }
            }
            
            println!("]");
        }
        OpCode::FfStore => {
            println!("FfStore");
        }
        OpCode::FfLoad => {
            println!("FfLoad");
        }
        OpCode::I64Load => {
            println!("I64Load");
        }
        OpCode::OpLt => {
            println!("OpLt");
        }
        OpCode::OpI64Mul => {
            println!("OpI64Mul");
        }
        OpCode::OpI64Lte => {
            println!("OpI64Lte");
        }
        OpCode::I64WrapFf => {
            println!("I64WrapFf");
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
            OpCode::StoreVariableI64 => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[component_tree.template_id].code, ip)?;
                let value = vm.pop_i64()?;
                vm.stack_i64[vm.base_pointer_i64 + var_idx] = Some(value);
            }
            OpCode::LoadVariableI64 => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(
                    &templates[component_tree.template_id].code, ip)?;
                let var = match vm.stack_i64.get(vm.base_pointer_i64 + var_idx) {
                    Some(v) => v,
                    None => return Err(Box::new(RuntimeError::StackOverflow)),
                };
                let var = match var {
                    Some(v) => v,
                    None => return Err(Box::new(RuntimeError::StackVariableIsNotSet)),
                };
                vm.push_i64(*var);
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
                        match signals[c.signals_start + sig_idx] {
                            Some(_) => {
                                return Err(Box::new(RuntimeError::SignalIsAlreadySet));
                            }
                            None => {
                                // println!("StoreCmpSignalAndRun, cmp_idx: {cmp_idx}, sig_idx: {sig_idx}, abs sig_idx: {}", c.signals_start + sig_idx);
                                signals[c.signals_start + sig_idx] = Some(value);
                            }
                        }
                        c.number_of_inputs -= 1;
                        execute(templates, signals, ff, c)?;
                    }
                }
            }
            OpCode::StoreCmpInput => {
                let sig_idx = vm.pop_usize()?;
                let cmp_idx = vm.pop_usize()?;
                let value = vm.pop_ff()?;
                match component_tree.components[cmp_idx] {
                    None => {
                        return Err(
                            Box::new(RuntimeError::UninitializedComponent))
                    }
                    Some(ref mut c) => {
                        match signals[c.signals_start + sig_idx] {
                            Some(_) => {
                                return Err(Box::new(RuntimeError::SignalIsAlreadySet));
                            }
                            None => {
                                // println!("StoreCmpInput, cmp_idx: {cmp_idx}, sig_idx: {sig_idx}, abs sig_idx: {}", c.signals_start + sig_idx);
                                signals[c.signals_start + sig_idx] = Some(value);
                            }
                        }
                    }
                }
            }
            OpCode::JumpIfFalseFf => {
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
            OpCode::JumpIfFalseI64 => {
                let offset_bytes = &templates[component_tree.template_id].code[ip..ip + size_of::<i32>()];
                let offset = i32::from_le_bytes((offset_bytes).try_into().unwrap());
                ip += size_of::<i32>();

                if vm.pop_i64()? == 0 {
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
            OpCode::OpI64Add => {
                let lhs = vm.pop_i64()?;
                let rhs = vm.pop_i64()?;
                vm.push_i64(lhs+rhs);
            }
            OpCode::OpI64Sub => {
                let lhs = vm.pop_i64()?;
                let rhs = vm.pop_i64()?;
                vm.push_i64(lhs-rhs);
            }
            OpCode::FfMReturn => {
                // TODO: Implement memory return operation
                // This would pop size, src, dst from stack and copy memory
                return Err(Box::new(RuntimeError::Assertion(-1)));
            }
            OpCode::FfMCall => {
                // TODO: Implement function call operation
                // This would read function index and arguments from bytecode
                // and execute the function
                return Err(Box::new(RuntimeError::Assertion(-2)));
            }
            OpCode::FfStore => {
                // TODO: Implement memory store operation
                return Err(Box::new(RuntimeError::Assertion(-3)));
            }
            OpCode::FfLoad => {
                // TODO: Implement memory load operation
                return Err(Box::new(RuntimeError::Assertion(-4)));
            }
            OpCode::I64Load => {
                // TODO: Implement memory load operation
                return Err(Box::new(RuntimeError::Assertion(-5)));
            }
            OpCode::OpLt => {
                // TODO: Implement field less-than comparison
                return Err(Box::new(RuntimeError::Assertion(-6)));
            }
            OpCode::OpI64Mul => {
                let lhs = vm.pop_i64()?;
                let rhs = vm.pop_i64()?;
                vm.push_i64(lhs * rhs);
            }
            OpCode::OpI64Lte => {
                let lhs = vm.pop_i64()?;
                let rhs = vm.pop_i64()?;
                vm.push_i64(if lhs <= rhs { 1 } else { 0 });
            }
            OpCode::I64WrapFf => {
                // TODO: Implement ff to i64 wrapping
                return Err(Box::new(RuntimeError::Assertion(-7)));
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