import regex as re
import heapq
from tqdm import tqdm
#lifting pretokenizer regex from tiktoken
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" 
merges_dict = {}

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Input
        input_path: str  Path to a text file with BPE tokenizer training data.
        vocab_size: int  A positive integer that defines the maximum final vocabulary size (including
        the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat
        them as hard boundaries that prevent merges across their spans, but do not include them when
        computing merge statistics.
        Your BPE training function should return the resulting vocabulary and merges:
    Output
        vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the
        vocabulary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list
        item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
        <token2>. The merges should be ordered by order of creation.
    """

    # special_tokens = ["<|endoftext|>", "<|padding|>", "<|mask|>"]
    # pattern = "|".join(special_tokens)
    # pattern = "|".join(re.escape(tok) for tok in special_tokens)

    # freq = {} # pretoken frequency map - find set of all unique pretokens and their frequencies
    # for chunk in process_chunks(input_path):
    #     for token in re.findall(PAT, re.sub(pattern," ",chunk.decode('utf-8'))):
    #         freq[token] = freq.get(token,0) + 1
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    freq = {}  # pretoken frequency map
    for chunk in process_chunks(input_path):
        text = chunk.decode('utf-8')
        segments = re.split(pattern, text) if pattern else [text]
        for segment in segments:
            for token in re.findall(PAT, segment):
                freq[token] = freq.get(token, 0) + 1


    pretoken_str, pretoken_freq = zip(*freq.items())
    pretokens = [list(pretoken.encode("utf-8")) for pretoken in pretoken_str]
    # once we have the frequency map - let's fix an index for each pretoken

    pair_map = {}
    for index in range(len(pretokens)):    
        token = pretokens[index]
        for i in range(len(token)-1):
            pair = (token[i],token[i+1])
            pair_map[pair] = pair_map.get(pair,0) + pretoken_freq[index]
            
    # create a priority queue to keep track of the most frequent pair
    pair_heap = [(-count,pair) for pair, count in pair_map.items()]
    heapq.heapify(pair_heap)

    # LAZY DELETION - pair_map is the source of truth
    # pair heap might have stale entries as pairs get deleted
    # instead of finding and deleting every single pair, we keep them and 
    # check if they are still alive by comparing the most frequent pair
    # to the pair_map, which is the source of truth. The point of not
    # deleting from the heap is to maintain the heap property.


    merges = []
    vocab = {index:bytes(resolve(index)) for index in range(256)} 
    for merge_index in (range(vocab_size - 256)):
        if len(pair_heap) == 0:
            break
        
        # LAZY DELETE: ensure top of the heap is valid((pair_heap[0][1] in pair_map)
        while not heap_entry_valid(pair_heap[0],pair_map):
            heapq.heappop(pair_heap)

        # find the highest frequency entry that is lexographically largest
        degenerate_stack = []
        highest_freq = pair_heap[0][0]
        while pair_heap[0][0] == highest_freq:
            val = heapq.heappop(pair_heap)
            if heap_entry_valid(val,pair_map):
                degenerate_stack.append(val)
        
        #lex_sort = sorted(degenerate_stack, key=lambda x: x[1])
        # lexographically largest is ill defined because the tokens may not correspond to 
        # valid unicode characters because the byte sequence is not necessarily utf decodable (could be incomplete)
        # instead of decoding to utf-8 and comparing, we compare the byte sequences instead. 
        
        lex_sort = sorted(degenerate_stack, key=lambda heap_entry: tuple(bytes(decode([index])) for index in heap_entry[1]))
        most_frequent_pair = lex_sort[-1] #grab the largest
        new_pair_index = 256 + merge_index
        merges_dict[new_pair_index] = most_frequent_pair[1]
        merges.append(tuple(bytes(resolve(p)) for p in most_frequent_pair[1]))
        vocab[new_pair_index] = bytes(resolve(new_pair_index))
        # push rest back into the heap
        for pair in lex_sort[:-1]:
            heapq.heappush(pair_heap,pair)
        
        # print ('most frequent pair ', most_frequent_pair[1],':', pair_map[most_frequent_pair[1]])
        pair_map_additive_diff = {}
        for i in range(len(pretokens)):
            pretokens[i], created, destroyed = mergebpairs(pretokens[i],most_frequent_pair[1],new_pair_index)
            for pair in created:
                pair_map_additive_diff[pair] = pair_map_additive_diff.get(pair,0) + pretoken_freq[i]
            # remove the destroyed pairs from pair_map 
            for pair in destroyed:
                if pair in pair_map:
                    pair_map[pair] -= pretoken_freq[i]
                if pair in pair_map and pair_map[pair] == 0:
                    del pair_map[pair]
    
        for new_pair in pair_map_additive_diff:
            updated_frequency = pair_map.get(new_pair,0) + pair_map_additive_diff[new_pair]
            pair_map[new_pair] = updated_frequency
            heapq.heappush(pair_heap,(-updated_frequency,new_pair))
    return vocab, merges


def process_chunks(path, chunksize = 2**16, special_token = "<|endoftext|>"):
    # reads a text file path points to and creates chunks of size chunksize - separated by special token 
    # returns an iterable of chunks of bytes
    # TODO: parallelize this.
    delimiter = special_token.encode("utf-8")
    # read bytes from the file
    with open(path,"rb") as f:
        buffer = b""
        while True:
            # read chunksize many bytes
            chunk = f.read(chunksize)
            if not chunk:
                # EOF - flush the buffer and exit the loop.
                yield buffer
                return
            # data after the last delimiter from previous chunk is saved in the buffer
            buffer += chunk
            # find the integer location of the latest delimiter
            id = buffer.rfind(delimiter)
            # if we can't find a delimiter - we keep going until we do.
            # this is in principle dangerous in case we dont'get a delimiter for very long.
            if id == -1:
                continue
            else:
                # if a delimiter is found, split it before the latest one 
                delimited = buffer[:id]
                # save the data after that and prepend it to the next chunk.
                buffer = buffer[id:]
                yield delimited

def mergebpairs(pretoken: list,pair: tuple, new_index: int):
    # given a pair we want to merge, it returns: 
    #   the pretoken with merges_dict applied, 
    #   pairs to be added, 
    #   pairs to be removed from the frequency map
    if not pair:
        return pretoken 
    else:
        to_add = []
        to_remove = []
        i = 0
        while i < len(pretoken):
            if i < len(pretoken) - 1 and pair[0] == pretoken[i] and pair[1]  == pretoken[i+1]:
                left = []
                right = []
                if i > 0: #has left
                    left = pretoken[:i]
                    to_remove.append((pretoken[i-1],pretoken[i]))
                    to_add.append((pretoken[i-1],new_index))
                if i + 2 < len(pretoken): #has right
                    right = pretoken[i+2:]
                    to_remove.append((pretoken[i+1],pretoken[i+2]))
                    to_add.append((new_index,pretoken[i+2]))
                to_remove.append(pair)
                pretoken = left + [new_index] + right
            i +=1
        return pretoken, to_add, to_remove

def heap_entry_valid(heap_entry: tuple, pair_map: dict):
    # check if the top node in the heap is stale
    freq = -heap_entry[0]
    pair = heap_entry[1]
    return pair in pair_map and pair_map[pair] == freq

def resolve(index: int):
    # given a token label - resolves it to an array from the base vocabulary
    if index < 256 or index not in merges_dict:
        return [index]
    else:
        return [item for sublist in [resolve(children) for children in merges_dict[index]] for item in sublist]

def decode(tokens: list[int]):
    # breaks down a list of tokens into base representation 
    return [bt for resolved in [resolve(index) for index in tokens] for bt in resolved]
