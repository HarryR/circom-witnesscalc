use crate::field::FieldOps;

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

fn read_instruction(code: &[u8], ip: usize) -> OpCode {
    unsafe { std::mem::transmute::<u8, OpCode>(code[ip]) }
}

pub enum RuntimeError {
    StackUnderflow,
    I32ToUsizeConversion,
    SignalIndexOutOfBounds,
    SignalIsNotSet,
}

pub fn execute<T>(
    templates: &[Template], signals: &mut [Option<T>],
    main_template_id: usize) -> Result<(), RuntimeError>
where
    T: FieldOps {

    let mut ip: usize = 0;
    let mut stack_ff: Vec<Option<T>> = Vec::new();
    let mut stack_i64: Vec<Option<i64>> = Vec::new();
    let mut stack_i64_base: usize = 0;

    'label: loop {
        if ip == templates[main_template_id].code.len() {
            break 'label;
        }

        let op_code = read_instruction(&templates[main_template_id].code, ip);
        ip += 1;

        match op_code {
            OpCode::NoOp => (),
            OpCode::LoadSignal => {
                let signal_idx = stack_i64.pop()
                    // Check that the stack is not empty
                    .ok_or_else(|| RuntimeError::StackUnderflow)?
                    // Checkout that the value in the stack is not null
                    .ok_or_else(|| RuntimeError::StackUnderflow)?;

                let signal_idx: usize = signal_idx.try_into()
                    .map_err(|_| RuntimeError::I32ToUsizeConversion)?;

                let s = signals.get(signal_idx)
                    .ok_or_else(|| RuntimeError::SignalIndexOutOfBounds)?
                    .ok_or_else(|| RuntimeError::SignalIsNotSet)?;
                
                stack_ff.push(Some(s));
            }
            OpCode::StoreSignal => {
                todo!();
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