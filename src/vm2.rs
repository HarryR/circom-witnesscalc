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
    //   1-byte argument type:
    //     0 = i64 literal
    //     1 = ff literal
    //     4-7 = ff.memory (bit flags: bit 0 = addr is variable, bit 1 = size is variable)
    //     8-11 = i64.memory (bit flags: bit 0 = addr is variable, bit 1 = size is variable)
    //   For literals: value bytes (8 for i64, T::BYTES for ff)
    //   For memory: 2 i64 values (either literal values or variable indices based on type flags)
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
    // Variable name mappings for debugging
    pub ff_variable_names: HashMap<usize, String>,
    pub i64_variable_names: HashMap<usize, String>,
}

fn read_instruction(code: &[u8], ip: usize) -> OpCode {
    unsafe { std::mem::transmute::<u8, OpCode>(code[ip]) }
}

// read 4 bytes from the code and return usize and the next instruction pointer
fn read_usize32(code: &[u8], ip: usize) -> (usize, usize) {
    let slice = code.get(ip..ip + 4)
        .expect("Code index out of bounds for usize32 read");
    let bytes: [u8; 4] = slice.try_into()
        .expect("Failed to convert slice to [u8; 4]");
    let v = u32::from_le_bytes(bytes) as usize;
    (v, ip + 4)
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
    #[error("Memory address is out of bounds")]
    MemoryAddressOutOfBounds,
    #[error("Value in the memory is None")]
    MemoryVariableIsNotSet,
    #[error("assertion: {0}")]
    Assertion(i64),
    #[error("Call stack overflow (max depth: 16384)")]
    CallStackOverflow,
    #[error("Call stack underflow")]
    CallStackUnderflow,
    #[error("Invalid function index: {0}")]
    InvalidFunctionIndex(usize),
    #[error("Unknown argument type in function call: {0}")]
    UnknownArgumentType(u8),
}

#[derive(Debug, Clone)]
enum ExecutionContext {
    Template,           // Executing template code
    Function(usize),    // Executing function code (function index)
}

#[derive(Debug)]
struct CallFrame {
    // Return execution context
    return_ip: usize,
    return_context: ExecutionContext,
    
    // Stack base pointers to restore
    return_stack_base_pointer_ff: usize,
    return_stack_base_pointer_i64: usize,
    
    // Memory base pointers to restore  
    return_memory_base_pointer_ff: usize,
    return_memory_base_pointer_i64: usize,
}

struct VM<T: FieldOps> {
    stack_ff: Vec<Option<T>>,
    stack_i64: Vec<Option<i64>>,
    stack_base_pointer_ff: usize,
    stack_base_pointer_i64: usize,
    memory_ff: Vec<Option<T>>,
    memory_i64: Vec<Option<i64>>,
    memory_base_pointer_ff: usize,
    memory_base_pointer_i64: usize,
    call_stack: Vec<CallFrame>,
    current_execution_context: ExecutionContext,
}

impl<T: FieldOps> VM<T> {
    fn new() -> Self {
        Self {
            stack_ff: Vec::new(),
            stack_i64: Vec::new(),
            stack_base_pointer_ff: 0,
            stack_base_pointer_i64: 0,
            memory_ff: vec![],
            memory_i64: vec![],
            memory_base_pointer_ff: 0,
            memory_base_pointer_i64: 0,
            call_stack: Vec::new(),
            current_execution_context: ExecutionContext::Template,
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

// Helper function to calculate the size of function arguments in bytecode
fn calculate_args_size<T: FieldOps>(code: &[u8], arg_count: u8) -> Result<usize, RuntimeError> {
    let mut offset = 0;
    for _ in 0..arg_count {
        if offset >= code.len() {
            return Err(RuntimeError::CodeIndexOutOfBounds);
        }
        
        let arg_type = code[offset];
        offset += 1;
        
        match arg_type {
            0 => offset += 8,  // i64 literal
            1 => offset += T::BYTES, // ff literal
            2 => offset += 8,  // ff variable
            3 => offset += 8,  // i64 variable
            4..=7 => offset += 16, // ff.memory (addr + size, both i64)
            8..=11 => offset += 16, // i64.memory (addr + size, both i64)
            _ => return Err(RuntimeError::CodeIndexOutOfBounds),
        }
    }
    Ok(offset)
}

// Helper function to process function arguments
fn process_function_arguments<T: FieldOps>(vm: &mut VM<T>, code: &[u8], arg_count: u8) -> Result<(), RuntimeError> {
    let mut offset = 0;
    let mut ff_arg_idx = 0;
    let mut i64_arg_idx = 0;
    
    for _ in 0..arg_count {
        if offset >= code.len() {
            return Err(RuntimeError::CodeIndexOutOfBounds);
        }
        
        let arg_type = code[offset];
        offset += 1;
        
        match arg_type {
            0 => { // i64 literal
                let value = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap());
                offset += 8;
                
                // Store in function's memory
                if vm.memory_i64.len() <= vm.memory_base_pointer_i64 + i64_arg_idx {
                    vm.memory_i64.resize(vm.memory_base_pointer_i64 + i64_arg_idx + 1, None);
                }
                vm.memory_i64[vm.memory_base_pointer_i64 + i64_arg_idx] = Some(value);
                i64_arg_idx += 1;
            }
            1 => { // ff literal  
                let value = T::from_le_bytes(&code[offset..offset+T::BYTES]).unwrap();
                offset += T::BYTES;
                
                // Store in function's memory
                if vm.memory_ff.len() <= vm.memory_base_pointer_ff + ff_arg_idx {
                    vm.memory_ff.resize(vm.memory_base_pointer_ff + ff_arg_idx + 1, None);
                }
                vm.memory_ff[vm.memory_base_pointer_ff + ff_arg_idx] = Some(value);
                ff_arg_idx += 1;
            }
            2 => { // ff variable
                let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                offset += 8;
                
                // Get caller's context from the call frame we just pushed
                let frame = vm.call_stack.last()
                    .ok_or(RuntimeError::CallStackUnderflow)?;
                let caller_stack_base = frame.return_stack_base_pointer_ff;
                
                // Load value from caller's ff variable stack
                let value = *vm.stack_ff.get(caller_stack_base + var_idx)
                    .and_then(|v| v.as_ref())
                    .ok_or(RuntimeError::StackVariableIsNotSet)?;
                
                // Store in function's memory
                if vm.memory_ff.len() <= vm.memory_base_pointer_ff + ff_arg_idx {
                    vm.memory_ff.resize(vm.memory_base_pointer_ff + ff_arg_idx + 1, None);
                }
                vm.memory_ff[vm.memory_base_pointer_ff + ff_arg_idx] = Some(value);
                ff_arg_idx += 1;
            }
            3 => { // i64 variable
                let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                offset += 8;
                
                // Get caller's context from the call frame we just pushed
                let frame = vm.call_stack.last()
                    .ok_or(RuntimeError::CallStackUnderflow)?;
                let caller_stack_base = frame.return_stack_base_pointer_i64;
                
                // Load value from caller's i64 variable stack
                let value = *vm.stack_i64.get(caller_stack_base + var_idx)
                    .and_then(|v| v.as_ref())
                    .ok_or(RuntimeError::StackVariableIsNotSet)?;
                
                // Store in function's memory
                if vm.memory_i64.len() <= vm.memory_base_pointer_i64 + i64_arg_idx {
                    vm.memory_i64.resize(vm.memory_base_pointer_i64 + i64_arg_idx + 1, None);
                }
                vm.memory_i64[vm.memory_base_pointer_i64 + i64_arg_idx] = Some(value);
                i64_arg_idx += 1;
            }
            4..=7 => { // ff.memory argument
                // Decode bit flags
                let addr_is_variable = (arg_type & 1) != 0;
                let size_is_variable = (arg_type & 2) != 0;
                
                // Get caller's context from the call frame we just pushed
                let frame = vm.call_stack.last()
                    .ok_or(RuntimeError::CallStackUnderflow)?;
                let caller_base_pointer_ff = frame.return_memory_base_pointer_ff;
                let caller_stack_base = frame.return_stack_base_pointer_i64;
                
                // Read and resolve address
                let src_addr = if addr_is_variable {
                    // It's a variable index - need to load from caller's stack
                    let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                    offset += 8;
                    
                    *vm.stack_i64.get(caller_stack_base + var_idx)
                        .and_then(|v| v.as_ref())
                        .ok_or(RuntimeError::StackVariableIsNotSet)? as usize
                } else {
                    // It's a literal value
                    let value = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap());
                    offset += 8;
                    value as usize
                };
                
                // Read and resolve size
                let size = if size_is_variable {
                    // It's a variable index - need to load from caller's stack
                    let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                    offset += 8;
                    
                    *vm.stack_i64.get(caller_stack_base + var_idx)
                        .and_then(|v| v.as_ref())
                        .ok_or(RuntimeError::StackVariableIsNotSet)? as usize
                } else {
                    // It's a literal value
                    let value = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap());
                    offset += 8;
                    value as usize
                };
                
                // Add caller's base pointer to source address
                let src_addr = src_addr + caller_base_pointer_ff;
                
                // Ensure source memory is valid
                if src_addr + size > vm.memory_ff.len() {
                    return Err(RuntimeError::MemoryAddressOutOfBounds);
                }
                
                // Copy from caller's memory to function's memory
                let dst_base = vm.memory_base_pointer_ff + ff_arg_idx;
                if vm.memory_ff.len() <= dst_base + size {
                    vm.memory_ff.resize(dst_base + size, None);
                }
                
                for i in 0..size {
                    vm.memory_ff[dst_base + i] = vm.memory_ff[src_addr + i];
                }
                
                ff_arg_idx += size;
            }
            8..=11 => { // i64.memory argument
                // Decode bit flags
                let addr_is_variable = (arg_type & 1) != 0;
                let size_is_variable = (arg_type & 2) != 0;
                
                // Get caller's context from the call frame we just pushed
                let frame = vm.call_stack.last()
                    .ok_or(RuntimeError::CallStackUnderflow)?;
                let caller_base_pointer_i64 = frame.return_memory_base_pointer_i64;
                let caller_stack_base = frame.return_stack_base_pointer_i64;
                
                // Read and resolve address
                let src_addr = if addr_is_variable {
                    // It's a variable index - need to load from caller's stack
                    let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                    offset += 8;
                    
                    *vm.stack_i64.get(caller_stack_base + var_idx)
                        .and_then(|v| v.as_ref())
                        .ok_or(RuntimeError::StackVariableIsNotSet)? as usize
                } else {
                    // It's a literal value
                    let value = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap());
                    offset += 8;
                    value as usize
                };
                
                // Read and resolve size
                let size = if size_is_variable {
                    // It's a variable index - need to load from caller's stack
                    let var_idx = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap()) as usize;
                    offset += 8;
                    
                    *vm.stack_i64.get(caller_stack_base + var_idx)
                        .and_then(|v| v.as_ref())
                        .ok_or(RuntimeError::StackVariableIsNotSet)? as usize
                } else {
                    // It's a literal value
                    let value = i64::from_le_bytes(code[offset..offset+8].try_into().unwrap());
                    offset += 8;
                    value as usize
                };
                
                // Add caller's base pointer to source address
                let src_addr = src_addr + caller_base_pointer_i64;
                
                // Ensure source memory is valid
                if src_addr + size > vm.memory_i64.len() {
                    return Err(RuntimeError::MemoryAddressOutOfBounds);
                }
                
                // Copy from caller's memory to function's memory
                let dst_base = vm.memory_base_pointer_i64 + i64_arg_idx;
                if vm.memory_i64.len() <= dst_base + size {
                    vm.memory_i64.resize(dst_base + size, None);
                }
                
                for i in 0..size {
                    vm.memory_i64[dst_base + i] = vm.memory_i64[src_addr + i];
                }
                
                i64_arg_idx += size;
            }
            _ => return Err(RuntimeError::UnknownArgumentType(arg_type)),
        }
    }
    
    Ok(())
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

pub fn disassemble_instruction_to_string<T>(
    code: &[u8], ip: usize, name: &str,
    ff_variable_names: &HashMap<usize, String>,
    i64_variable_names: &HashMap<usize, String>) -> (usize, String)
where
    T: FieldOps {

    let mut output = format!("{:08x} [{:10}] ", ip, name);

    let op_code = read_instruction(code, ip);
    let mut ip = ip + 1usize;

    match op_code {
        OpCode::NoOp => {
            output.push_str("NoOp");
        }
        OpCode::LoadSignal => {
            output.push_str("LoadSignal");
        }
        OpCode::StoreSignal => {
            output.push_str("StoreSignal");
        }
        OpCode::PushI64 => {
            let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
            ip += size_of::<i64>();
            output.push_str(&format!("PushI64: {}", v));
        }
        OpCode::PushFf => {
            let s = &code[ip..ip+T::BYTES];
            ip += T::BYTES;
            let v = T::from_le_bytes(s).unwrap();
            output.push_str(&format!("PushFf: {}", v));
        }
        OpCode::StoreVariableFf => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            let var_name = ff_variable_names.get(&var_idx)
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            output.push_str(&format!("StoreVariableFf: {}{}", var_idx, var_name));
        }
        OpCode::StoreVariableI64 => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            let var_name = i64_variable_names.get(&var_idx)
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            output.push_str(&format!("StoreVariableI64: {}{}", var_idx, var_name));
        }
        OpCode::LoadVariableI64 => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            let var_name = i64_variable_names.get(&var_idx)
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            output.push_str(&format!("LoadVariableI64: {}{}", var_idx, var_name));
        }
        OpCode::LoadVariableFf => {
            let var_idx: usize;
            (var_idx, ip) = usize_from_code(code, ip).unwrap();
            let var_name = ff_variable_names.get(&var_idx)
                .map(|s| format!(" ({})", s))
                .unwrap_or_default();
            output.push_str(&format!("LoadVariableFf: {}{}", var_idx, var_name));
        }
        OpCode::JumpIfFalseFf => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            let newIP = if v < 0 {
                ip - (v.unsigned_abs() as usize)
            } else {
                ip + (v as usize)
            };
            output.push_str(&format!("JumpIfFalseFf: {:+} -> {:08x}", v, newIP));
        }
        OpCode::JumpIfFalseI64 => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            let newIP = if v < 0 {
                ip - (v.unsigned_abs() as usize)
            } else {
                ip + (v as usize)
            };
            output.push_str(&format!("JumpIfFalseI64: {:+} -> {:08x}", v, newIP));
        }
        OpCode::Jump => {
            let v = i32::from_le_bytes((&code[ip..ip+size_of::<i32>()]).try_into().unwrap());
            ip += size_of::<i32>();
            let newIP = if v < 0 {
                ip - (v.unsigned_abs() as usize)
            } else {
                ip + (v as usize)
            };
            output.push_str(&format!("Jump: {:+} -> {:08x}", v, newIP));
        }
        OpCode::LoadCmpSignal => {
            output.push_str("LoadCmpSignal");
        }
        OpCode::StoreCmpSignalAndRun => {
            output.push_str("StoreCmpSignalAndRun");
        }
        OpCode::StoreCmpInput => {
            output.push_str("StoreCmpInput");
        }
        OpCode::OpMul => {
            output.push_str("OpMul");
        }
        OpCode::OpAdd => {
            output.push_str("OpAdd");
        }
        OpCode::OpNeq => {
            output.push_str("OpNeq");
        }
        OpCode::OpDiv => {
            output.push_str("OpDiv");
        }
        OpCode::OpSub => {
            output.push_str("OpSub");
        }
        OpCode::OpEq => {
            output.push_str("OpEq");
        }
        OpCode::OpEqz => {
            output.push_str("OpEqz");
        }
        OpCode::OpI64Add => {
            output.push_str("OpI64Add");
        }
        OpCode::OpI64Sub => {
            output.push_str("OpI64Sub");
        }
        OpCode::Error => {
            output.push_str("Error");
        }
        OpCode::FfMReturn => {
            output.push_str("FfMReturn");
        }
        OpCode::FfMCall => {
            // Read function index
            let func_idx = u32::from_le_bytes((&code[ip..ip+4]).try_into().unwrap());
            ip += 4;
            
            // Read argument count
            let arg_count = code[ip];
            ip += 1;
            
            output.push_str(&format!("FfMCall: func_idx={}, args=[", func_idx));
            
            // Parse each argument
            for i in 0..arg_count {
                if i > 0 {
                    output.push_str(", ");
                }
                
                let arg_type = code[ip];
                ip += 1;
                
                match arg_type {
                    0 => { // i64 literal
                        let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        output.push_str(&format!("i64.{}", v));
                    }
                    1 => { // ff literal
                        let v = T::from_le_bytes(&code[ip..ip+T::BYTES]).unwrap();
                        ip += T::BYTES;
                        output.push_str(&format!("ff.{}", v));
                    }
                    2 => { // ff variable
                        let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        output.push_str(&format!("ff.var[{}]", v));
                    }
                    3 => { // i64 variable
                        let v = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        output.push_str(&format!("i64.var[{}]", v));
                    }
                    4..=7 => { // ff memory
                        let addr_is_variable = (arg_type & 1) != 0;
                        let size_is_variable = (arg_type & 2) != 0;
                        
                        let addr_val = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        let size_val = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        
                        output.push_str("ff.memory(");
                        if addr_is_variable {
                            output.push_str(&format!("var[{}]", addr_val));
                        } else {
                            output.push_str(&format!("{}", addr_val));
                        }
                        output.push(',');
                        if size_is_variable {
                            output.push_str(&format!("var[{}]", size_val));
                        } else {
                            output.push_str(&format!("{}", size_val));
                        }
                        output.push(')');
                    }
                    8..=11 => { // i64 memory
                        let addr_is_variable = (arg_type & 1) != 0;
                        let size_is_variable = (arg_type & 2) != 0;
                        
                        let addr_val = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        let size_val = i64::from_le_bytes((&code[ip..ip+8]).try_into().unwrap());
                        ip += 8;
                        
                        output.push_str("i64.memory(");
                        if addr_is_variable {
                            output.push_str(&format!("var[{}]", addr_val));
                        } else {
                            output.push_str(&format!("{}", addr_val));
                        }
                        output.push(',');
                        if size_is_variable {
                            output.push_str(&format!("var[{}]", size_val));
                        } else {
                            output.push_str(&format!("{}", size_val));
                        }
                        output.push(')');
                    }
                    _ => {
                        output.push_str(&format!("unknown_arg_type({})", arg_type));
                    }
                }
            }
            
            output.push(']');
        }
        OpCode::FfStore => {
            output.push_str("FfStore");
        }
        OpCode::FfLoad => {
            output.push_str("FfLoad");
        }
        OpCode::I64Load => {
            output.push_str("I64Load");
        }
        OpCode::OpLt => {
            output.push_str("OpLt");
        }
        OpCode::OpI64Mul => {
            output.push_str("OpI64Mul");
        }
        OpCode::OpI64Lte => {
            output.push_str("OpI64Lte");
        }
        OpCode::I64WrapFf => {
            output.push_str("I64WrapFf");
        }
    }

    (ip, output)
}

pub fn disassemble_instruction<T>(
    code: &[u8], ip: usize, name: &str,
    ff_variable_names: &HashMap<usize, String>,
    i64_variable_names: &HashMap<usize, String>) -> usize
where
    T: FieldOps {
    let (new_ip, output) = disassemble_instruction_to_string::<T>(
        code, ip, name, ff_variable_names, i64_variable_names);
    println!("{}", output);
    new_ip
}

// Helper function to get the currently executing template/function
fn get_current_template<'a, T: FieldOps>(
    vm: &VM<T>,
    circuit: &'a Circuit<T>,
    component_tree: &Component,
) -> &'a Template {
    match vm.current_execution_context {
        ExecutionContext::Template => &circuit.templates[component_tree.template_id],
        ExecutionContext::Function(func_idx) => &circuit.functions[func_idx],
    }
}

pub fn execute<F, T: FieldOps>(
    circuit: &Circuit<T>, signals: &mut [Option<T>], ff: &F,
    component_tree: &mut Component) -> Result<(), Box<dyn Error>>
where
    for <'a> &'a F: FieldOperations<Type = T> {

    let mut ip: usize = 0;
    let mut vm = VM::<T>::new();
    
    // Initialize with template's variable counts (function calls will resize as needed)
    vm.stack_ff.resize_with(
        circuit.templates[component_tree.template_id].vars_ff_num, || None);
    vm.stack_i64.resize_with(
        circuit.templates[component_tree.template_id].vars_i64_num, || None);

    // Cache current template reference - only update when execution context changes
    let mut current_template = get_current_template(&vm, circuit, component_tree);

    'label: loop {
        if ip == current_template.code.len() {
            // Handle end of current execution context
            match vm.current_execution_context {
                ExecutionContext::Template => {
                    // Template completed normally
                    break 'label;
                }
                ExecutionContext::Function(_) => {
                    // Function ended without explicit return - this is an error
                    return Err(Box::new(RuntimeError::Assertion(-998))); // Function didn't return
                }
            }
        }

        disassemble_instruction::<T>(
            &current_template.code, ip, &current_template.name,
            &current_template.ff_variable_names,
            &current_template.i64_variable_names);

        let op_code = read_instruction(&current_template.code, ip);
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
                        (&current_template.code[ip..ip+8])
                            .try_into().unwrap()));
                ip += 8;
            }
            OpCode::PushFf => {
                let s = &current_template.code[ip..ip+T::BYTES];
                ip += T::BYTES;
                let v = ff.parse_le_bytes(s)?;
                vm.push_ff(v);
            }
            OpCode::StoreVariableFf => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(&current_template.code, ip)?;
                let value = vm.pop_ff()?;
                vm.stack_ff[vm.stack_base_pointer_ff + var_idx] = Some(value);
            }
            OpCode::StoreVariableI64 => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(&current_template.code, ip)?;
                let value = vm.pop_i64()?;
                vm.stack_i64[vm.stack_base_pointer_i64 + var_idx] = Some(value);
            }
            OpCode::LoadVariableI64 => {
                let var_idx: usize;
                (var_idx, ip) = usize_from_code(&current_template.code, ip)?;
                let var = match vm.stack_i64.get(vm.stack_base_pointer_i64 + var_idx) {
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
                (var_idx, ip) = usize_from_code(&current_template.code, ip)?;
                let var = match vm.stack_ff.get(vm.stack_base_pointer_ff + var_idx) {
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
                        execute(circuit, signals, ff, c)?;
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
                let offset_bytes = &current_template.code[ip..ip + size_of::<i32>()];
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
                let offset_bytes = &current_template.code[ip..ip + size_of::<i32>()];
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
                let offset_bytes = &current_template.code[ip..ip + size_of::<i32>()];
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
                // Pop size, src, dst from stack
                let size = vm.pop_usize()?;
                let src_addr = vm.pop_usize()?;
                let dst_addr = vm.pop_usize()?;
                
                // Pop call frame to get return context
                let call_frame = vm.call_stack.pop()
                    .ok_or(RuntimeError::CallStackUnderflow)?;
                
                eprintln!("DEBUG FfMReturn: current ip={:08x}, return_ip={:08x}", ip, call_frame.return_ip);
                eprintln!("DEBUG FfMReturn: current context={:?}, return context={:?}", vm.current_execution_context, call_frame.return_context);
                
                // Copy memory from function's space to caller's space
                for i in 0..size {
                    let src_idx = src_addr + i + vm.memory_base_pointer_ff;
                    let dst_idx = dst_addr + i + call_frame.return_memory_base_pointer_ff;
                    
                    if src_idx < vm.memory_ff.len() {
                        if dst_idx >= vm.memory_ff.len() {
                            vm.memory_ff.resize(dst_idx + 1, None);
                        }
                        vm.memory_ff[dst_idx] = vm.memory_ff[src_idx];
                        // Clean up function memory
                        vm.memory_ff[src_idx] = None;
                    }
                }
                
                // Restore execution context
                ip = call_frame.return_ip;
                vm.current_execution_context = call_frame.return_context;
                vm.stack_base_pointer_ff = call_frame.return_stack_base_pointer_ff;
                vm.stack_base_pointer_i64 = call_frame.return_stack_base_pointer_i64;
                vm.memory_base_pointer_ff = call_frame.return_memory_base_pointer_ff;
                vm.memory_base_pointer_i64 = call_frame.return_memory_base_pointer_i64;
                
                // Switch back to caller's execution context
                current_template = get_current_template(&vm, circuit, component_tree);
                eprintln!("DEBUG FfMReturn: after restoration ip={:08x}, template={}", ip, current_template.name);
            }
            OpCode::FfMCall => {
                // Check call stack depth
                if vm.call_stack.len() >= 16384 {
                    return Err(Box::new(RuntimeError::CallStackOverflow));
                }
                
                let func_idx: usize;
                (func_idx, ip) = read_usize32(&current_template.code, ip);

                // Validate function index
                if func_idx >= circuit.functions.len() {
                    return Err(Box::new(RuntimeError::InvalidFunctionIndex(func_idx)));
                }

                // Read argument count
                let arg_count = current_template.code[ip];
                ip += 1;
                
                // Create call frame
                let call_frame = CallFrame {
                    return_ip: ip + calculate_args_size::<T>(&current_template.code[ip..], arg_count)?,
                    return_context: vm.current_execution_context.clone(),
                    return_stack_base_pointer_ff: vm.stack_base_pointer_ff,
                    return_stack_base_pointer_i64: vm.stack_base_pointer_i64,
                    return_memory_base_pointer_ff: vm.memory_base_pointer_ff,
                    return_memory_base_pointer_i64: vm.memory_base_pointer_i64,
                };
                vm.call_stack.push(call_frame);
                
                // Set up new execution context
                vm.current_execution_context = ExecutionContext::Function(func_idx);
                vm.stack_base_pointer_ff = vm.stack_ff.len();
                vm.stack_base_pointer_i64 = vm.stack_i64.len();
                vm.memory_base_pointer_ff = vm.memory_ff.len();
                vm.memory_base_pointer_i64 = vm.memory_i64.len();
                
                // Allocate space for function's local variables
                vm.stack_ff.resize(vm.stack_base_pointer_ff + circuit.functions[func_idx].vars_ff_num, None);
                vm.stack_i64.resize(vm.stack_base_pointer_i64 + circuit.functions[func_idx].vars_i64_num, None);
                
                // Process arguments and copy to function memory
                process_function_arguments(&mut vm, &current_template.code[ip..], arg_count)?;
                
                // Switch to function execution context
                current_template = get_current_template(&vm, circuit, component_tree);
                ip = 0; // Start executing function from beginning
            }
            OpCode::FfStore => {
                let addr: usize = vm.pop_i64()?.try_into()
                    .map_err(|_| Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                let addr = addr.checked_add(vm.memory_base_pointer_ff)
                    .ok_or(Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                if addr >= vm.memory_ff.len() {
                    vm.memory_ff.resize(addr + 1, None);
                }
                let value = vm.pop_ff()?;
                vm.memory_ff[addr] = Some(value);
            }
            OpCode::FfLoad => {
                let addr: usize = vm.pop_i64()?.try_into()
                    .map_err(|_| Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                let addr = addr.checked_add(vm.memory_base_pointer_ff)
                    .ok_or(Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                if addr >= vm.memory_ff.len() {
                    return Err(Box::new(RuntimeError::MemoryAddressOutOfBounds));
                }
                let value = vm.memory_ff.get(addr)
                    .and_then(|v| v.as_ref())
                    .ok_or(RuntimeError::MemoryVariableIsNotSet)?;
                vm.push_ff(*value);
            }
            OpCode::I64Load => {
                let addr: usize = vm.pop_i64()?.try_into()
                    .map_err(|_| Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                let addr = addr.checked_add(vm.memory_base_pointer_i64)
                    .ok_or(Box::new(RuntimeError::MemoryAddressOutOfBounds))?;
                if addr >= vm.memory_i64.len() {
                    return Err(Box::new(RuntimeError::MemoryAddressOutOfBounds));
                }
                let value = vm.memory_i64.get(addr)
                    .and_then(|v| v.as_ref())
                    .ok_or(RuntimeError::MemoryVariableIsNotSet)?;
                vm.push_i64(*value);
            }
            OpCode::OpLt => {
                let rhs = vm.pop_ff()?;
                let lhs = vm.pop_ff()?;
                let result = ff.lt(lhs, rhs);
                vm.push_ff(result);
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
                let ff_val = vm.pop_ff()?;
                // Convert field element to i64 by taking lower 64 bits
                // This matches the behavior expected by i64.wrap_ff
                let bytes = ff_val.to_le_bytes();
                let i64_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
                let i64_val = i64::from_le_bytes(i64_bytes);
                vm.push_i64(i64_val);
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