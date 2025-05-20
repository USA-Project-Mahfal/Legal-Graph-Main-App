import re
import pandas as pd
from tqdm import tqdm

def optimized_hybrid_chunking(docs_df, semantic_min_size=150, semantic_max_size=400,
                             hierarchical_levels=[400, 800], position_importance=True):
    """
    Optimized hybrid chunking combining semantic boundaries with hierarchical representation
    for both search and graph neural network applications.
    """
    chunks = []
    chunk_id_counter = 0 # Use a single counter

    section_patterns = [
        r'(?i)(?:\n|\s{2,})(?:section|article)\s+\d+[\.\:]\s+[A-Z]',
        r'\n[A-Z][A-Z\s]+(?:\n|\s{2,})',
        r'(?:\n|\s{2,})(?:\d+\.\d+|\d+\.)\s+[A-Z]',
        r'(?i)(?:\n|\s{2,})(?:DEFINITIONS|DEFINED TERMS)(?:\n|\s{2,})',
        r'(?:\n|\s{2,})(?:WHEREAS:|IN WITNESS WHEREOF:|NOW, THEREFORE,)',
        r'(?i)(?:\n|\s{2,})(?:RECITALS|WITNESSETH|APPENDIX|EXHIBIT|SCHEDULE|ANNEX)\s+[A-Za-z0-9]'
    ]
    compiled_patterns = [re.compile(pattern) for pattern in section_patterns]

    # Read all text files from the chunk_input folders
    import os
    import glob

    data = []
    id_counter = 1
    
    folders = ['License_Agreements', 'Maintenance', 'Service', 'Sponsorship', 'Strategic Alliance']
    
    for folder in folders:
        folder_path = os.path.join('chunk_input', folder)
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        for file_path in txt_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                
            file_name = os.path.basename(file_path)
            name = os.path.splitext(file_name)[0]
            
            data.append({
                'id': id_counter,
                'text': text_content,
                'name': name,
                'category': folder
            })
            id_counter += 1
    
    # Create DataFrame from all files
    docs_df = pd.DataFrame(data)
    docs_df.to_csv('docs_df.csv', index=False)

    for _, doc in tqdm(docs_df.iterrows(), total=len(docs_df), desc="Creating hybrid chunks"):
        doc_id_val = doc['id']
        text = doc['text']
        
        # Create word to character offset mapping for the current document
        doc_text_words = text.split() # Based on how preprocess_legal_text works
        char_start_offsets_for_words = [0] * len(doc_text_words)
        current_char_offset = 0
        # This simple split & find might be fragile if text has complex whitespace.
        # preprocess_legal_text aims to normalize this.
        temp_text_for_offsets = text
        for i, word in enumerate(doc_text_words):
            try:
                word_pos = temp_text_for_offsets.find(word)
                char_start_offsets_for_words[i] = current_char_offset + word_pos
                advance_offset = word_pos + len(word)
                current_char_offset += advance_offset
                temp_text_for_offsets = temp_text_for_offsets[advance_offset:]
            except Exception: # Fallback if find fails unexpectedly
                 if i > 0: char_start_offsets_for_words[i] = char_start_offsets_for_words[i-1] + len(doc_text_words[i-1]) + 1
                 else: char_start_offsets_for_words[i] = 0


        doc_chunks_meta_l1 = [] # Store L1 chunk metadata for this doc

        # STEP 1: Create semantic boundaries
        boundaries = [0]
        for pattern in compiled_patterns:
            for match in pattern.finditer(text):
                boundaries.append(match.start())
        paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        boundaries.extend(paragraph_breaks)
        boundaries.append(len(text))
        boundaries = sorted(list(set(boundaries)))

        # STEP 2: Create initial semantic_chunks (L1 precursor)
        raw_semantic_chunks = []
        special_sections = {'definitions': None, 'termination': None, 'confidentiality': None, 'indemnification': None}
        
        for i in range(len(boundaries) - 1):
            chunk_text = text[boundaries[i]:boundaries[i+1]].strip()
            if not chunk_text: continue

            lower_text = chunk_text.lower()
            for section_type_key in special_sections:
                if section_type_key in lower_text and len(chunk_text.split()) > 20: # Arbitrary length
                    if special_sections[section_type_key] is None: # Mark first occurrence
                       special_sections[section_type_key] = len(raw_semantic_chunks) 

            total_initial_chunks = len(boundaries) - 1
            position_val = "middle"
            position_score_val = 0.8 if position_importance else 0.7
            if total_initial_chunks > 0 : # Avoid division by zero
                if i < total_initial_chunks * 0.2:
                    position_val = "beginning"; position_score_val = 0.95 if position_importance else 0.7
                elif i > total_initial_chunks * 0.8:
                    position_val = "end"; position_score_val = 0.7 if position_importance else 0.7
            
            raw_semantic_chunks.append({
                'text': chunk_text, 'start_idx': boundaries[i], 'end_idx': boundaries[i+1],
                'position': position_val, 'position_score': position_score_val, 'original_idx': i
            })

        # STEP 3: Process L1 chunk sizes
        processed_l1_chunks = []
        idx_raw_sem = 0
        while idx_raw_sem < len(raw_semantic_chunks):
            chunk = raw_semantic_chunks[idx_raw_sem]
            words = chunk['text'].split()

            if len(words) < semantic_min_size and idx_raw_sem < len(raw_semantic_chunks) - 1:
                next_chunk = raw_semantic_chunks[idx_raw_sem + 1]
                combined_text = chunk['text'] + " " + next_chunk['text'] # Simple space join
                if len(combined_text.split()) <= semantic_max_size:
                    # Merge chunk with next_chunk
                    raw_semantic_chunks[idx_raw_sem + 1]['text'] = combined_text
                    raw_semantic_chunks[idx_raw_sem + 1]['start_idx'] = chunk['start_idx']
                    # Position/score of the merged chunk could be re-evaluated or taken from the first.
                    # For simplicity, next_chunk's original position info is largely kept, but start_idx is updated.
                    idx_raw_sem += 1 # Skip current chunk as it's merged into next
                    continue
            
            if len(words) > semantic_max_size:
                # Try natural sub-divisions first
                sub_texts_natural = re.split(r'\n\s*\n', chunk['text'])
                if len(sub_texts_natural) > 1 and all(len(t.split()) < semantic_max_size for t in sub_texts_natural if t.strip()):
                    current_char_offset_in_chunk = 0
                    for sub_idx, sub_text_natural in enumerate(sub_texts_natural):
                        sub_text_natural = sub_text_natural.strip()
                        if not sub_text_natural: continue
                        
                        sub_pos = chunk['position']
                        sub_score = chunk['position_score']
                        if sub_idx > 0 : sub_score = max(0.6, chunk['position_score'] - 0.1 * sub_idx)

                        processed_l1_chunks.append({
                            'text': sub_text_natural,
                            'start_idx': chunk['start_idx'] + current_char_offset_in_chunk,
                            'end_idx': chunk['start_idx'] + current_char_offset_in_chunk + len(sub_text_natural),
                            'position': sub_pos, 'position_score': sub_score, 
                            'parent_original_idx': chunk.get('original_idx', -1)
                        })
                        current_char_offset_in_chunk += len(sub_text_natural) + (len(chunk['text']) - current_char_offset_in_chunk - len(sub_text_natural) > 0) # Approx for separator
                else: # Forced splits
                    current_word_idx_in_chunk = 0
                    chunk_words = chunk['text'].split() # Words of the current large L1 chunk
                    
                    # Create char offsets for words within this specific chunk['text']
                    offsets_in_chunk_text = [0] * len(chunk_words)
                    temp_chunk_text_ptr = 0
                    search_text_segment = chunk['text']
                    for i_cw, cw in enumerate(chunk_words):
                        try:
                            pos_cw = search_text_segment.find(cw)
                            offsets_in_chunk_text[i_cw] = temp_chunk_text_ptr + pos_cw
                            adv = pos_cw + len(cw)
                            temp_chunk_text_ptr += adv
                            search_text_segment = search_text_segment[adv:]
                        except: # Fallback
                            if i_cw > 0: offsets_in_chunk_text[i_cw] = offsets_in_chunk_text[i_cw-1] + len(chunk_words[i_cw-1]) + 1
                            else: offsets_in_chunk_text[i_cw] = 0


                    while current_word_idx_in_chunk < len(chunk_words):
                        sub_chunk_words = chunk_words[current_word_idx_in_chunk : current_word_idx_in_chunk + semantic_max_size]
                        if not sub_chunk_words: break
                        
                        sub_text_forced = " ".join(sub_chunk_words) # Reconstruct with single spaces

                        # Determine char start/end for this sub_text_forced within original document
                        sub_chunk_char_start_in_chunktext = offsets_in_chunk_text[current_word_idx_in_chunk]
                        
                        # End is start of last word + len of last word
                        idx_of_last_word_in_sub_chunk = current_word_idx_in_chunk + len(sub_chunk_words) - 1
                        sub_chunk_char_end_in_chunktext = offsets_in_chunk_text[idx_of_last_word_in_sub_chunk] + len(chunk_words[idx_of_last_word_in_sub_chunk])


                        sub_pos = chunk['position']
                        sub_score = chunk['position_score']
                        if current_word_idx_in_chunk > 0 : sub_score = max(0.6, chunk['position_score'] - 0.1)

                        processed_l1_chunks.append({
                            'text': sub_text_forced,
                            'start_idx': chunk['start_idx'] + sub_chunk_char_start_in_chunktext,
                            'end_idx': chunk['start_idx'] + sub_chunk_char_end_in_chunktext,
                            'position': sub_pos, 'position_score': sub_score,
                            'parent_original_idx': chunk.get('original_idx', -1)
                        })
                        current_word_idx_in_chunk += len(sub_chunk_words)
            else: # Chunk size is acceptable
                processed_l1_chunks.append(chunk)
            idx_raw_sem += 1
            
        # STEP 4: Create final L1 chunks and then hierarchical chunks (L2+)
        doc_all_level_chunks = []

        # Add L1 (semantic) chunks
        for l1_chunk_data in processed_l1_chunks:
            is_special = False
            special_type_name = None
            parent_orig_idx = l1_chunk_data.get('parent_original_idx', l1_chunk_data.get('original_idx'))

            for sec_type, orig_idx in special_sections.items():
                if orig_idx is not None and parent_orig_idx == orig_idx:
                    is_special = True
                    special_type_name = sec_type
                    break
            
            current_chunk_id = chunk_id_counter
            final_l1_chunk = {
                'chunk_id': current_chunk_id, 'doc_id': doc_id_val, 'doc_name': doc['name'],
                'category': doc['category'], 'text': l1_chunk_data['text'],
                'chunk_method': 'semantic', 'level': 'L1',
                'start_idx': l1_chunk_data['start_idx'], 'end_idx': l1_chunk_data['end_idx'],
                'document_position': l1_chunk_data['position'], 'position_score': l1_chunk_data['position_score'],
                'is_special_section': is_special
            }
            if is_special: final_l1_chunk['section_type'] = special_type_name
            
            doc_all_level_chunks.append(final_l1_chunk)
            doc_chunks_meta_l1.append({ # For L2+ contained_chunks logic
                'chunk_id': current_chunk_id,
                'char_start_idx': l1_chunk_data['start_idx'], # Char idx
                'char_end_idx': l1_chunk_data['end_idx']      # Char idx
            })
            chunk_id_counter += 1

        # Add L2+ (hierarchical) chunks
        for level_idx, hier_chunk_size_words in enumerate(hierarchical_levels):
            level_name = f"L{level_idx + 2}"
            if len(doc_text_words) < hier_chunk_size_words * 1.5: continue # Skip if doc too short for this level

            overlap_words = min(hier_chunk_size_words // 4, 50)
            
            for i_word_hier in range(0, len(doc_text_words), hier_chunk_size_words - overlap_words):
                hier_sub_words = doc_text_words[i_word_hier : i_word_hier + hier_chunk_size_words]
                if len(hier_sub_words) < hier_chunk_size_words // 3 : continue

                hier_text = " ".join(hier_sub_words)
                
                # Hierarchical chunk boundaries in characters
                hier_char_start = char_start_offsets_for_words[i_word_hier]
                idx_last_word_in_hier = i_word_hier + len(hier_sub_words) -1
                hier_char_end = char_start_offsets_for_words[idx_last_word_in_hier] + len(doc_text_words[idx_last_word_in_hier]) if idx_last_word_in_hier < len(doc_text_words) else len(text)


                pos_hier = "middle"; score_hier = 0.75 if position_importance else 0.7
                total_hier_chunks_at_level = max(1, (len(doc_text_words) - hier_chunk_size_words) // (hier_chunk_size_words - overlap_words) +1)
                current_hier_chunk_index = i_word_hier // (hier_chunk_size_words-overlap_words)
                if total_hier_chunks_at_level > 0:
                    if current_hier_chunk_index < total_hier_chunks_at_level / 3:
                        pos_hier = "beginning"; score_hier = 0.85 if position_importance else 0.7
                    elif current_hier_chunk_index > 2 * total_hier_chunks_at_level / 3:
                        pos_hier = "end"; score_hier = 0.7 if position_importance else 0.7
                
                contained_l1_ids = []
                for l1_meta in doc_chunks_meta_l1:
                    # Check for overlap: max(start1, start2) < min(end1, end2)
                    if max(hier_char_start, l1_meta['char_start_idx']) < min(hier_char_end, l1_meta['char_end_idx']):
                        contained_l1_ids.append(l1_meta['chunk_id'])
                
                current_chunk_id = chunk_id_counter
                doc_all_level_chunks.append({
                    'chunk_id': current_chunk_id, 'doc_id': doc_id_val, 'doc_name': doc['name'],
                    'category': doc['category'], 'text': hier_text,
                    'chunk_method': 'hierarchical', 'level': level_name, 'level_size': hier_chunk_size_words,
                    'start_idx': hier_char_start, # Store char indices for consistency if preferred
                    'end_idx': hier_char_end,     # Or store word indices i_word_hier, i_word_hier + len(hier_sub_words)
                    'document_position': pos_hier, 'position_score': score_hier,
                    'contained_chunks': contained_l1_ids # List of L1 chunk IDs
                })
                chunk_id_counter += 1
        
        # Full document chunk for small documents
        # Consider adjusting this threshold if you want to change how medium-sized documents are handled
        if len(doc_text_words) < 1000: # Arbitrary threshold, potentially increase if desired
            current_chunk_id = chunk_id_counter
            doc_all_level_chunks.append({
                'chunk_id': current_chunk_id, 'doc_id': doc_id_val, 'doc_name': doc['name'],
                'category': doc['category'], 'text': text,
                'chunk_method': 'full_document', 'level': 'full',
                'start_idx': 0, 'end_idx': len(text),
                'document_position': 'complete', 'position_score': 1.0,
                'contained_chunks': [m['chunk_id'] for m in doc_chunks_meta_l1]
            })
            chunk_id_counter +=1
            
        chunks.extend(doc_all_level_chunks)

    chunks_df = pd.DataFrame(chunks)
    
    if not chunks_df.empty:
        # Add chunk_relationships
        # This can be slow on very large dataframes.
        # Pre-calculating lookups or using groupby could optimize if needed.
        
        # Create a lookup for L1 chunks per document
        l1_chunks_by_doc = {}
        if 'level' in chunks_df.columns and 'doc_id' in chunks_df.columns and 'chunk_id' in chunks_df.columns:
             l1_chunks_by_doc = chunks_df[chunks_df['level'] == 'L1'].groupby('doc_id')['chunk_id'].apply(list).to_dict()

        # Create a lookup for hierarchical parents
        # A L1 chunk's parents are L2+ chunks that contain it.
        hier_parents_lookup = {} # Key: L1_chunk_id, Value: list of L2+_chunk_ids
        if 'contained_chunks' in chunks_df.columns and 'chunk_method' in chunks_df.columns:
            for _, row in chunks_df.iterrows():
                if row['chunk_method'] == 'hierarchical' or row['chunk_method'] == 'full_document':
                    if isinstance(row['contained_chunks'], list):
                        for l1_child_id in row['contained_chunks']:
                            if l1_child_id not in hier_parents_lookup:
                                hier_parents_lookup[l1_child_id] = []
                            hier_parents_lookup[l1_child_id].append(row['chunk_id'])
        
        def get_relationships(row):
            rels = {'same_doc_l1_chunks': [], 'hierarchical_parents': []}
            if row['level'] == 'L1': # Relationships primarily defined for L1 nodes
                # Same document L1 chunks (excluding self)
                rels['same_doc_l1_chunks'] = [cid for cid in l1_chunks_by_doc.get(row['doc_id'], []) if cid != row['chunk_id']]
                # Hierarchical parents
                rels['hierarchical_parents'] = hier_parents_lookup.get(row['chunk_id'], [])
            return rels
            
        chunks_df['chunk_relationships'] = chunks_df.apply(get_relationships, axis=1)

    print(f"Created {len(chunks_df)} hybrid chunks from {len(docs_df)} documents")
    return chunks_df

# Call the function and save the resulting DataFrame
if __name__ == "__main__":
    # Create an empty DataFrame to pass to the function
    initial_docs_df = pd.DataFrame()
    
    # Call the optimized_hybrid_chunking function
    result_df = optimized_hybrid_chunking(initial_docs_df)

    # Save the resulting DataFrame to a CSV file
    result_df.to_csv('hybrid_chunks.csv', index=False)
