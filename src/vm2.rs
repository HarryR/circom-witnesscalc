use crate::field::FieldOps;

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

fn read_instruction(code: &[u8], ip: usize) -> OpCode {
    unsafe { std::mem::transmute::<u8, OpCode>(code[ip]) }
}

pub enum RuntimeError {
    StackUnderflow,
    StackVariableIsNotSet,
    I32ToUsizeConversion,
    SignalIndexOutOfBounds,
    SignalIsNotSet,
    SignalIsAlreadySet,
}

struct VM<T: FieldOps> {
    stack_ff: Vec<Option<T>>,
    stack_i64: Vec<Option<i64>>,
}

impl<T: FieldOps> VM<T> {
    fn new() -> Self {
        Self {
            stack_ff: Vec::new(),
            stack_i64: Vec::new(),
        }
    }

    fn push_ff(&mut self, value: Option<T>) {
        self.stack_ff.push(value);
    }

    fn pop_ff(&mut self) -> Result<T, RuntimeError> {
        self.stack_ff.pop().ok_or(RuntimeError::StackUnderflow)?
            .ok_or(RuntimeError::StackVariableIsNotSet)
    }

    fn push_i64(&mut self, value: Option<i64>) {
        self.stack_i64.push(value);
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

pub fn execute<T>(
    templates: &[Template], signals: &mut [Option<T>],
    main_template_id: usize) -> Result<(), RuntimeError>
where
    T: FieldOps {

    let mut ip: usize = 0;
    let mut vm = VM::<T>::new();

    'label: loop {
        if ip == templates[main_template_id].code.len() {
            break 'label;
        }

        let op_code = read_instruction(&templates[main_template_id].code, ip);
        ip += 1;

        match op_code {
            OpCode::NoOp => (),
            OpCode::LoadSignal => {
                let signal_idx = vm.pop_usize()?;
                let s = signals.get(signal_idx)
                    .ok_or_else(|| RuntimeError::SignalIndexOutOfBounds)?
                    .ok_or_else(|| RuntimeError::SignalIsNotSet)?;

                vm.push_ff(Some(s));
            }
            OpCode::StoreSignal => {
                let signal_idx = vm.pop_usize()?;
                if signal_idx >= signals.len() {
                    return Err(RuntimeError::SignalIndexOutOfBounds);
                }
                if signals[signal_idx].is_some() {
                    return Err(RuntimeError::SignalIsAlreadySet);
                }
                signals[signal_idx] = Some(vm.pop_ff()?);
            }
            OpCode::PushI64 => {
                todo!();
            }
            OpCode::PushFf => {
                todo!();
            }
            OpCode::StoreVariable => {
                todo!();
            }
            OpCode::LoadVariableI64 => {
                todo!();
            }
            OpCode::LoadVariableFf => {
                todo!();
            }
            OpCode::OpMul => {
                todo!();
            }
            OpCode::OpAdd => {
                todo!();
            }
        }
    }

    Ok(())
}