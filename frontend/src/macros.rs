#[macro_export]
macro_rules! advance_single_token {
    ($self:expr, $token_type:expr) => {{
        $self.chars.next();
        $self.current += 1;
        let span = $self.start..$self.current;
        $self.start = $self.current;
        return Ok($crate::span::Spanned::new($token_type, span));
    }};
}

#[macro_export]
macro_rules! handle_operator {
    ($self:expr, $first_char:expr, $second_char:expr, 
     $single_token:expr, $double_token:expr) => {{
        $self.chars.next(); 
        $self.current += 1;
        
        if $self.chars.peek() == Some(&$second_char) {
            $self.chars.next(); 
            $self.current += 1;
            
            let span = $self.start..$self.current;
            $self.start = $self.current;
            return Ok($crate::span::Spanned::new($double_token, span));
        } else {
            let span = $self.start..$self.current;
            $self.start = $self.current;
            return Ok($crate::span::Spanned::new($single_token, span));
        }
    }};
}
