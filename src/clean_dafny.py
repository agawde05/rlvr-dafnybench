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
    // Noa Leron 207131871
// Tsuri Farhana 315016907



predicate Sorted(q: seq<int>) {
	forall i,j :: 0 <= i <= j < |q| ==> q[i] <= q[j]
}

/*
Goal: Implement the well known merge sort algorithm in O(a.Length X log_2(a.Length)) time, recursively.

- Divide the contents of the original array into two local arrays
- After sorting the local arrays (recursively), merge the contents of the two returned arrays using the Merge method (see below)
- DO NOT modify the specification or any other part of the method's signature
- DO NOT introduce any further methods
*/
method MergeSort(a: array<int>) returns (b: array<int>)
	ensures b.Length == a.Length && Sorted(b[..]) && multiset(a[..]) == multiset(b[..])
{
	if (a.Length <= 1) {b := a;}
    else{
        var mid: nat := a.Length / 2;
        var a1: array<int> := new int[mid];
        var a2: array<int> := new int[a.Length - mid];

        var i: nat := 0;
        while (i < a1.Length )
        {
            a1[i] := a[i];
            a2[i] := a[i+mid];
            i:=i+1;
        }
        
        if(a1.Length < a2.Length) {
            a2[i] := a[i+mid];
        } // If a.Length is odd.
        else{
        }

        a1:= MergeSort(a1);
        a2:= MergeSort(a2);
        b := new int [a.Length];
        Merge(b, a1, a2);
    }
} 

ghost predicate Inv(a: seq<int>, a1: seq<int>, a2: seq<int>, i: nat, mid: nat){
    (i <= |a1|) && (i <= |a2|) && (i+mid <= |a|) &&
    (a1[..i] == a[..i]) && (a2[..i] == a[mid..(i+mid)])
}

/*
Goal: Implement iteratively, correctly, efficiently, clearly

DO NOT modify the specification or any other part of the method's signature
*/
method Merge(b: array<int>, c: array<int>, d: array<int>)
	requires b != c && b != d && b.Length == c.Length + d.Length
	requires Sorted(c[..]) && Sorted(d[..])
	ensures Sorted(b[..]) && multiset(b[..]) == multiset(c[..])+multiset(d[..])
	modifies b
{
	var i: nat, j: nat := 0, 0;
	while i + j < b.Length
	{	
		i,j := MergeLoop (b,c,d,i,j);
	}
	LemmaMultysetsEquals(b[..],c[..],d[..],i,j);	
		
}


//This is a method that replace the loop body
method {:verify true} MergeLoop(b: array<int>, c: array<int>, d: array<int>,i0: nat , j0: nat)  returns (i: nat, j: nat)
		requires b != c && b != d && b.Length == c.Length + d.Length
		requires Sorted(c[..]) && Sorted(d[..])
		requires i0 <= c.Length && j0 <= d.Length && i0 + j0 <= b.Length
		requires InvSubSet(b[..],c[..],d[..],i0,j0)
		requires InvSorted(b[..],c[..],d[..],i0,j0)
		requires i0 + j0 < b.Length

		modifies b

		ensures i <= c.Length && j <= d.Length && i + j <= b.Length
		ensures InvSubSet(b[..],c[..],d[..],i,j)
		ensures InvSorted(b[..],c[..],d[..],i,j)
		//decreases ensures
		ensures 0 <= c.Length - i < c.Length - i0 || (c.Length - i == c.Length - i0 && 0 <= d.Length - j < d.Length - j0)
		{

			i,j := i0,j0;
				
				if(i == c.Length || (j< d.Length && d[j] < c[i])){
					// in this case we take the next value from d
				b[i+j] := d[j];
				lemmaInvSubsetTakeValueFromD(b[..],c[..],d[..],i,j);

				j := j + 1;
			}
			else{
					// in this case we take the next value from c
				
				b[i+j] := c[i];

				lemmaInvSubsetTakeValueFromC(b[..],c[..],d[..],i,j);
				i := i + 1;
			}


		}

	
//Loop invariant - b is sprted so far and the next two potential values that will go into b are bigger then the biggest value in b.
ghost predicate InvSorted(b: seq<int>, c: seq<int>, d: seq<int>, i: nat, j: nat){
	i <= |c| && j <= |d| && i + j <= |b| &&
	((i+j > 0 && i < |c|) ==> (b[j + i - 1] <= c[i])) &&
	((i+j > 0 && j < |d|) ==> (b[j + i - 1] <= d[j])) &&
	Sorted(b[..i+j])
	}


//Loop invariant - the multiset of the prefix of b so far is the same multiset as the prefixes of c and d so far.
ghost predicate InvSubSet(b: seq<int>, c: seq<int>, d: seq<int>, i: nat, j: nat){
	i <= |c| && j <= |d| && i + j <= |b| &&
	multiset(b[..i+j]) == multiset(c[..i]) + multiset(d[..j])
}

//This lemma helps dafny see that if the prefixs of arrays are the same multiset until the end of the arrays,
//all the arrays are the same multiset.
lemma LemmaMultysetsEquals (b: seq<int>, c: seq<int>, d: seq<int>, i: nat, j: nat)
	requires i == |c|;
	requires j == |d|;
	requires i + j == |b|;
	requires multiset(b[..i+j]) == multiset(c[..i]) + multiset(d[..j])
	ensures multiset(b[..]) == multiset(c[..])+multiset(d[..]);
	{
	}


//This lemma helps dafny see that after adding the next value from c to b the prefixes are still the same subsets.
lemma lemmaInvSubsetTakeValueFromC (b: seq<int>, c: seq<int>, d: seq<int>, i: nat, j: nat)
	requires i < |c|;
	requires j <= |d|;
	requires i + j < |b|;
	requires |c| + |d| == |b|;
	requires multiset(b[..i+j]) == multiset(c[..i]) + multiset(d[..j])
	requires b[i+j] == c[i]
	ensures multiset(b[..i+j+1]) == multiset(c[..i+1])+multiset(d[..j])
	{
	}



//This lemma helps dafny see that after adding the next value from d to b the prefixes are still the same subsets.
lemma{:verify true} lemmaInvSubsetTakeValueFromD (b: seq<int>, c: seq<int>, d: seq<int>, i: nat, j: nat)
	requires i <= |c|;
	requires j < |d|;
	requires i + j < |b|;
	requires |c| + |d| == |b|;
	requires multiset(b[..i+j]) == multiset(c[..i]) + multiset(d[..j])
	requires b[i+j] == d[j]
	ensures multiset(b[..i+j+1]) == multiset(c[..i])+multiset(d[..j+1])
	{
	}





method Main() {
	var a := new int[3] [4, 8, 6];
	var q0 := a[..];
	a := MergeSort(a);
	print "\nThe sorted version of ", q0, " is ", a[..];

	a := new int[5] [3, 8, 5, -1, 10];
	q0 := a[..];
	a := MergeSort(a);
	print "\nThe sorted version of ", q0, " is ", a[..];
	//assert a[..] == [-1, 3, 5, 8, 10];
}
    """
    cleaned_code = remove_code_from_method(test_code)
    print(cleaned_code)