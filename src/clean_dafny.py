def remove_code_from_method(old_code: str) -> str:
    """
    Remove executable bodies from Dafny methods while preserving function bodies.
    See docstring in specification for full details.
    """
    METHOD_LIKE_KEYWORDS = ("method", "lemma", "constructor")
    
    n = len(old_code)
    result = []
    i = 0
    
    def skip_line_comment(pos):
        """Skip from // to end of line"""
        while pos < n and old_code[pos] != '\n':
            pos += 1
        return pos
    
    def skip_block_comment(pos):
        """Skip from /* to */"""
        pos += 2
        while pos < n - 1 and old_code[pos:pos+2] != '*/':
            pos += 1
        return pos + 2 if pos < n - 1 else n
    
    def skip_string(pos):
        """Skip a double-quoted string"""
        pos += 1
        while pos < n and old_code[pos] != '"':
            if old_code[pos] == '\\' and pos + 1 < n:
                pos += 2
            else:
                pos += 1
        return pos + 1 if pos < n else n
    
    def skip_char_literal(pos):
        """Skip a single-quoted character literal"""
        pos += 1
        while pos < n and old_code[pos] != "'":
            if old_code[pos] == '\\' and pos + 1 < n:
                pos += 2
            else:
                pos += 1
        return pos + 1 if pos < n else n
    
    def is_word_boundary_before(pos):
        """Check if position is at a word boundary"""
        if pos == 0:
            return True
        prev = old_code[pos - 1]
        return not (prev.isalnum() or prev == '_')
    
    def is_word_boundary_after(pos):
        """Check if position after keyword is at a word boundary"""
        if pos >= n:
            return True
        next_char = old_code[pos]
        # Word boundary: whitespace, '(', '<', or any non-identifier char
        return not (next_char.isalnum() or next_char == '_')
    
    def find_body_open_brace(pos):
        """Find the opening '{' of a method body starting from pos"""
        paren_depth = 0
        while pos < n:
            if old_code[pos:pos+2] == '//':
                pos = skip_line_comment(pos)
            elif old_code[pos:pos+2] == '/*':
                pos = skip_block_comment(pos)
            elif old_code[pos] == '"':
                pos = skip_string(pos)
            elif old_code[pos] == "'":
                pos = skip_char_literal(pos)
            elif old_code[pos] == '(':
                paren_depth += 1
                pos += 1
            elif old_code[pos] == ')':
                paren_depth -= 1
                pos += 1
            elif old_code[pos] == '{' and paren_depth == 0:
                # Check if this is an annotation {:...} - if so, skip it
                if pos + 1 < n and old_code[pos + 1] == ':':
                    # Skip the annotation block
                    depth = 1
                    pos += 1
                    while pos < n and depth > 0:
                        if old_code[pos] == '{':
                            depth += 1
                        elif old_code[pos] == '}':
                            depth -= 1
                        pos += 1
                else:
                    return pos
            else:
                pos += 1
        return -1
    
    def find_matching_close_brace(pos):
        """Find the matching '}' for '{' at pos"""
        depth = 1
        pos += 1
        while pos < n and depth > 0:
            if old_code[pos:pos+2] == '//':
                pos = skip_line_comment(pos)
            elif old_code[pos:pos+2] == '/*':
                pos = skip_block_comment(pos)
            elif old_code[pos] == '"':
                pos = skip_string(pos)
            elif old_code[pos] == "'":
                pos = skip_char_literal(pos)
            elif old_code[pos] == '{':
                depth += 1
                pos += 1
            elif old_code[pos] == '}':
                depth -= 1
                pos += 1
            else:
                pos += 1
        return pos - 1 if depth == 0 else -1
    
    def get_line_indent(pos):
        """Get the indentation of the line containing pos"""
        line_start = pos
        while line_start > 0 and old_code[line_start - 1] != '\n':
            line_start -= 1
        indent = []
        k = line_start
        while k < pos and old_code[k] in ' \t':
            indent.append(old_code[k])
            k += 1
        return ''.join(indent)
    
    while i < n:
        # Skip comments in main loop - copy them as-is
        if old_code[i:i+2] == '//':
            start = i
            i = skip_line_comment(i)
            result.append(old_code[start:i])
            continue
        
        if old_code[i:i+2] == '/*':
            start = i
            i = skip_block_comment(i)
            result.append(old_code[start:i])
            continue
        
        # Skip strings - copy them as-is
        if old_code[i] == '"':
            start = i
            i = skip_string(i)
            result.append(old_code[start:i])
            continue
        
        if old_code[i] == "'":
            start = i
            i = skip_char_literal(i)
            result.append(old_code[start:i])
            continue
        
        # Check for method-like keywords
        found_kw = None
        for kw in METHOD_LIKE_KEYWORDS:
            kw_len = len(kw)
            if (old_code[i:i+kw_len] == kw and 
                is_word_boundary_before(i) and 
                is_word_boundary_after(i + kw_len)):
                found_kw = kw
                break
        
        if found_kw:
            # Process method-like declaration
            method_start = i
            open_brace = find_body_open_brace(i + len(found_kw))
            
            if open_brace != -1:
                close_brace = find_matching_close_brace(open_brace)
                
                if close_brace != -1:
                    # Output: everything from method_start to open_brace (inclusive)
                    result.append(old_code[method_start:open_brace + 1])
                    result.append('\n')
                    # Add indentation and closing brace
                    indent = get_line_indent(method_start)
                    result.append(indent + '}')
                    
                    i = close_brace + 1
                    continue
        
        # Default: copy character
        result.append(old_code[i])
        i += 1
    
    return ''.join(result)


if __name__ == "__main__":
    test_code = """

method FindWinner'<Candidate(==)>(a: seq<Candidate>, ghost K: Candidate) returns (k: Candidate)
  requires HasMajority(a, 0, |a|, K) // K has a (strict) majority of the votes
  ensures k == K  // find K
{
  k := a[0]; // Current candidate: the first element
  var lo, up, c := 0, 1, 1; // Window: [0..1], number of occurrences of k in the window: 1
  while up < |a|
  {
    if a[up] == k {
      // One more occurrence of k
      up, c := up + 1, c + 1;
    } else if 2 * c > up + 1 - lo {
      // An occurrence of another value, but k still has the majority
      up := up + 1;
    } else {
      // An occurrence of another value and k just lost the majority.
      // Prove that k has exactly 50% in the future window a[lo..up + 1]:
      calc /* k has 50% among a[lo..up + 1] */ {
        true;
      ==  // negation of the previous branch condition;
        2 * c <= up + 1 - lo;
      ==  // loop invariant (3)
        2 * Count(a, lo, up, k) <= up + 1 - lo;
      == calc {
           true;
         ==  // loop invariant (2)
           HasMajority(a, lo, up, k);
         ==  // def. HasMajority
           2 * Count(a, lo, up, k) > up - lo;
         ==
           2 * Count(a, lo, up, k) >= up + 1 - lo;
         }
        2 * Count(a, lo, up, k) == up + 1 - lo;
      }
      up := up + 1;

      // We are going to start a new window a[up..up + 1] and choose a new candidate,
      // so invariants (2) and (3) will be easy to re-establish.
      // To re-establish (1) we have to prove that K has majority among a[up..], as up will become the new lo.
      // The main idea is that we had enough K's in a[lo..], and there cannot be too many in a[lo..up].
      calc /* K has majority among a[up..] */ {
        2 * Count(a, up, |a|, K);
      ==  { Lemma_Split(a, lo, up, |a|, K); }
        2 * Count(a, lo, |a|, K) - 2 * Count(a, lo, up, K);
      >  { assert HasMajority(a, lo, |a|, K); } // loop invariant (1)
        |a| - lo - 2 * Count(a, lo, up, K);
      >=  { if k == K {
              calc {
                2 * Count(a, lo, up, K);
              ==
                2 * Count(a, lo, up, k);
              ==  { assert 2 * Count(a, lo, up, k) == up - lo; } // k has 50% among a[lo..up]
                up - lo;
              }
            } else {
              calc {
                2 * Count(a, lo, up, K);
              <=  { Lemma_Unique(a, lo, up, k, K); }
                2 * ((up - lo) - Count(a, lo, up, k));
              ==  { assert 2 * Count(a, lo, up, k) == up - lo; } // k has 50% among a[lo..up]
                up - lo;
              }
            }
          }
        |a| - lo - (up - lo);
      ==
        |a| - up;
      }

      k, lo, up, c := a[up], up, up + 1, 1;
    }
  }
  Lemma_Unique(a, lo, |a|, K, k);  // both k and K have a majority among a[lo..], so K == k
}


    """
    cleaned_code = remove_code_from_method(test_code)
    print(cleaned_code)