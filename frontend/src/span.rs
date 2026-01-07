use std::fmt;
use std::hash::{Hash, Hasher};

pub type Span = std::ops::Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T: Hash> Hash for Spanned<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
    }
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    pub fn default(node: T) -> Self {
        Self::new(node, DUMMY_SPAN)
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }

    pub fn inner(&self) -> &T {
        &self.node
    }

    pub fn into_inner(self) -> T {
        self.node
    }
}

impl<T: fmt::Display> fmt::Display for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.node.fmt(f)
    }
}

/// Extension trait for working with spans.
pub trait SpanExt {
    /// Merge two spans into one that covers both.
    fn merge(&self, other: &Span) -> Span;
}

impl SpanExt for Span {
    fn merge(&self, other: &Span) -> Span {
        let start = self.start.min(other.start);
        let end = self.end.max(other.end);
        start..end
    }
}

pub const DUMMY_SPAN: Span = 0..0;

/// Stores the source text for error reporting.
#[derive(Debug, Clone)]
pub struct SourceMap {
    source: String,
    /// Byte offsets of line starts (0-indexed).
    line_starts: Vec<usize>,
}

impl SourceMap {
    /// Create a new source map from source text.
    pub fn new(source: String) -> Self {
        let mut line_starts = vec![0];
        for (i, c) in source.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self {
            source,
            line_starts,
        }
    }

    /// Get the source text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Convert a byte offset to a (line, column) pair (1-indexed).
    pub fn offset_to_line_col(&self, offset: usize) -> (usize, usize) {
        let line = self
            .line_starts
            .iter()
            .rposition(|&start| start <= offset)
            .unwrap_or(0);
        let col = offset - self.line_starts[line];
        (line + 1, col + 1)
    }

    /// Get the text of a specific line (1-indexed).
    pub fn get_line(&self, line: usize) -> Option<&str> {
        if line == 0 || line > self.line_starts.len() {
            return None;
        }
        let start = self.line_starts[line - 1];
        let end = self
            .line_starts
            .get(line)
            .copied()
            .unwrap_or(self.source.len());
        // Trim the trailing newline if present
        let text = &self.source[start..end];
        Some(text.trim_end_matches('\n').trim_end_matches('\r'))
    }

    /// Get the text covered by a span.
    pub fn span_text(&self, span: &Span) -> &str {
        &self.source[span.clone()]
    }

    /// Format a span as a location string (e.g., "5:12").
    pub fn format_location(&self, span: &Span) -> String {
        let (line, col) = self.offset_to_line_col(span.start);
        format!("{}:{}", line, col)
    }

    /// Format an error with source context, showing the problematic span underlined.
    ///
    /// Example output:
    /// ```text
    /// error: type mismatch
    ///  --> 5:12
    ///   |
    /// 5 |     x := foo()
    ///   |          ^^^^^ expected `int`, found `unit`
    /// ```
    pub fn format_error(&self, span: &Span, message: &str, label: Option<&str>) -> String {
        if span == &DUMMY_SPAN {
            return format!("error: {}", message);
        }

        let (line, col) = self.offset_to_line_col(span.start);
        let line_text = self.get_line(line).unwrap_or("");

        // Calculate underline length
        let (end_line, _) = self.offset_to_line_col(span.end.saturating_sub(1).max(span.start));
        let underline_len = if line == end_line {
            span.end.saturating_sub(span.start).max(1)
        } else {
            // Multi-line span: underline to end of first line
            line_text.len().saturating_sub(col - 1).max(1)
        };

        let line_num_width = format!("{}", line).len();
        let padding = " ".repeat(line_num_width);

        let mut output = String::new();
        output.push_str(&format!("error: {}\n", message));
        output.push_str(&format!("{}--> {}:{}\n", padding, line, col));
        output.push_str(&format!("{} |\n", padding));
        output.push_str(&format!("{} | {}\n", line, line_text));
        output.push_str(&format!(
            "{} | {}{}",
            padding,
            " ".repeat(col - 1),
            "^".repeat(underline_len)
        ));

        if let Some(label) = label {
            output.push_str(&format!(" {}", label));
        }
        output.push('\n');

        output
    }
}

/// Trait for types that can produce formatted error messages.
#[allow(unused)]
pub trait Diagnostic {
    /// Get the span of the error.
    fn span(&self) -> &Span;

    /// Get the main error message.
    fn message(&self) -> String;

    /// Get an optional label for the underlined span.
    fn label(&self) -> Option<String> {
        None
    }

    /// Format the error with source context.
    fn format(&self, source_map: &SourceMap) -> String {
        source_map.format_error(self.span(), &self.message(), self.label().as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_merge() {
        let a = 5..10;
        let b = 8..15;
        assert_eq!(a.merge(&b), 5..15);

        let c = 0..5;
        let d = 10..20;
        assert_eq!(c.merge(&d), 0..20);
    }

    #[test]
    fn test_spanned() {
        let s: Spanned<i32> = Spanned::new(42, 0..2);
        assert_eq!(s.inner(), &42);
        assert_eq!(s.span, 0..2);

        let mapped = s.map(|x| x * 2);
        assert_eq!(mapped.inner(), &84);
        assert_eq!(mapped.span, 0..2);
    }

    #[test]
    fn test_source_map_line_col() {
        let source = "hello\nworld\nfoo".to_string();
        let sm = SourceMap::new(source);

        assert_eq!(sm.offset_to_line_col(0), (1, 1)); // 'h'
        assert_eq!(sm.offset_to_line_col(5), (1, 6)); // '\n'
        assert_eq!(sm.offset_to_line_col(6), (2, 1)); // 'w'
        assert_eq!(sm.offset_to_line_col(11), (2, 6)); // '\n'
        assert_eq!(sm.offset_to_line_col(12), (3, 1)); // 'f'
    }

    #[test]
    fn test_source_map_get_line() {
        let source = "hello\nworld\nfoo".to_string();
        let sm = SourceMap::new(source);

        assert_eq!(sm.get_line(1), Some("hello"));
        assert_eq!(sm.get_line(2), Some("world"));
        assert_eq!(sm.get_line(3), Some("foo"));
        assert_eq!(sm.get_line(0), None);
        assert_eq!(sm.get_line(4), None);
    }

    #[test]
    fn test_source_map_span_text() {
        let source = "hello world".to_string();
        let sm = SourceMap::new(source);

        assert_eq!(sm.span_text(&(0..5)), "hello");
        assert_eq!(sm.span_text(&(6..11)), "world");
    }
}
