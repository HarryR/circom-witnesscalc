#[repr(u8)]
#[derive(Debug)]
pub enum OpCode {
    NoOp = 0,
    // Put signals to the stack
    // required stack_i64: signal index
    LoadSignal      = 1,
    StoreSignal     = 2,
    PushI64         = 3, // Push i64 value to the stack
    PushFf          = 4, // Push i64 value to the stack
    // Set variables from the stack
    // arguments:      offset from the base pointer
    // required stack: values to store equal to variables number from arguments
    StoreVariable   = 5,
    LoadVariableI64 = 6,
    LoadVariableFf  = 7,
    OpMul           = 8,
    OpAdd           = 9,
}

pub struct Template {
    pub name: String,
    pub code: Vec<u8>,
}