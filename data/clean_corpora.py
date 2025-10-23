import os
import glob
import json
import re


def check_missing_fields(jsonl_file):

    fields_to_check = ["original_text", "lemmatized_text", "UPOS", "XPOS", "feature"]
    

    missing_counts = {field: 0 for field in fields_to_check}
    

    updated_sentences = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            sentence = json.loads(line)
            missing_info = []
            

            for field in fields_to_check:
                if not sentence.get(field):
                    missing_counts[field] += 1
                    missing_info.append(f"miss/{field}")

            if missing_info:
                sentence["clean"] = ", ".join(missing_info)
            
            updated_sentences.append(sentence)
    

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for sentence in updated_sentences:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")

    print(f"✅ `{jsonl_file}` summary of missing fields：")
    for field, count in missing_counts.items():
        print(f"   - {field} misses in {count} lines")






def remove_duplicates(jsonl_file):

    unique_sentences = {}
    duplicate_count = 0
    updated_sentences = []


    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            sentence = json.loads(line)
            key = (sentence["newdoc_id"], sentence["sent_id"])

            if key in unique_sentences:

                sentence["clean"] = "repetition"
                duplicate_count += 1
            else:
                unique_sentences[key] = sentence

            updated_sentences.append(sentence)


    output_clean_file = jsonl_file.replace(".jsonl", "_clean.jsonl")
    with open(output_clean_file, "w", encoding="utf-8") as f:
        for sentence in updated_sentences:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")


    print(f"✅ Before deduplication: {len(updated_sentences)} sentences")
    print(f"✅ After deduplication: {len(unique_sentences)} sentences")
    print(f"⚠️ Found {duplicate_count} duplicate sentences (marked as 'repetition')")

def replace_gaps_and_update_clean(input_path, output_path):

    updated_data = []
    gap_count_1 = 0
    gap_count_2 = 0

    dots_inside_brackets_pattern = re.compile(r"\[.*?\]|\[\.*\]|\[\s\]|\[\u2026\]|\[\s*\]")

    def replace_gaps_in_text(text):

        if not isinstance(text, str):
            return text, False, False

        modified_1 = False
        modified_2 = False

        tokens = text.split()
        new_tokens = []

        for token in tokens:
            if "----" in token:
                new_tokens.append("[gap]")
                modified_1 = True
            elif dots_inside_brackets_pattern.search(token):
                new_tokens.append("[gap]")
                modified_2 = True
            else:
                new_tokens.append(token)

        return " ".join(new_tokens), modified_1, modified_2

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())


            original_text, modified_1_ot, modified_2_ot = replace_gaps_in_text(item.get("original_text", ""))
            item["original_text"] = original_text


            lemmatized_text, modified_1_lt, modified_2_lt = replace_gaps_in_text(item.get("lemmatized_text", ""))
            item["lemmatized_text"] = lemmatized_text


            modified_1 = modified_1_ot or modified_1_lt
            modified_2 = modified_2_ot or modified_2_lt

            if modified_1:
                gap_count_1 += 1
            if modified_2:
                gap_count_2 += 1


            if modified_1 or modified_2:
                if "clean" in item and isinstance(item["clean"], str):
                    if item["clean"] == "clean":
                        item["clean"] = []
                    elif isinstance(item["clean"], str):
                        item["clean"] = [item["clean"]]

                    if modified_1:
                        item["clean"].append("destruction/----")
                    if modified_2:
                        item["clean"].append("destruction/[...]")

                    item["clean"] = "; ".join(set(item["clean"]))

                else:
                    item["clean"] = "destruction/----" if modified_1 else "destruction/[...]"

            updated_data.append(item)


    with open(output_path, "w", encoding="utf-8") as f:
        for item in updated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ accomplished: {output_path}")
    print(f"📊 statistics:")
    print(f"   - {gap_count_1} lines contained '----' and were replaced with '[gap]'.")
    print(f"   - {gap_count_2} lines contained '[...]' and entire tokens were replaced with '[gap]'.")
    print(f"   - Total modified lines: {gap_count_1 + gap_count_2}")




def replace_gap_dots_asterisk_unknown_words(jsonl_path):

    gap_pattern_double_dot = re.compile(r".*\.\..*")
    gap_pattern_ellipsis = re.compile(r".*….*")
    gap_pattern_asterisk = re.compile(r".*\*.*")
    # gap_pattern_question = re.compile(r".*\?.*")
    modified_sentences = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in lines:
            item = json.loads(line.strip())
            modified = False

            for field in ["original_text", "lemmatized_text"]:
                if field in item:
                    words = item[field].split()
                    processed_words = []

                    for word in words:
                        if gap_pattern_double_dot.search(word) or gap_pattern_ellipsis.search(
                                word) or gap_pattern_asterisk.search(word):
                            processed_words.append("[gap]")
                            modified = True
                        elif field == "lemmatized_text" and (word == "UNKNOWN" or word == "." or word == "·"):
                            processed_words.append("[gap]")
                            modified = True
                        elif field == "original_text" and (word == "." or word == "·"):
                            processed_words.append("[gap]")
                            modified = True
                        else:
                            processed_words.append(word)

                    item[field] = " ".join(processed_words)

            if modified:
                modified_sentences += 1
                if "clean" in item:
                    if item["clean"] == "clean":
                        item["clean"] = "destruction/.."
                    else:
                        item["clean"] += "; destruction/.."

            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ 处理完成，修改了 {modified_sentences} 个 JSONL 句子！")

def remove_noise_signs(input_path):

    noise_pattern = re.compile(r"[·Ⲽⲽ\(\)]")
    cleaned_count = 0

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()


    modified_lines = []

    for line in lines:
        item = json.loads(line.strip())
        modified = False


        original_text_before = item.get("original_text", "")
        lemmatized_text_before = item.get("lemmatized_text", "")

        if isinstance(original_text_before, str) and noise_pattern.search(original_text_before):
            modified = True
            item["original_text"] = noise_pattern.sub("", original_text_before)


        if isinstance(lemmatized_text_before, str) and noise_pattern.search(lemmatized_text_before):
            modified = True
            item["lemmatized_text"] = noise_pattern.sub("", lemmatized_text_before)


        if modified:
            cleaned_count += 1
            print("\n🔍 **Before Cleaning:**")
            print(f"   Original Text: {original_text_before}")
            print(f"   Lemmatized Text: {lemmatized_text_before}")
            print("✅ **After Cleaning:**")
            print(f"   Original Text: {item['original_text']}")
            print(f"   Lemmatized Text: {item['lemmatized_text']}")


        modified_lines.append(json.dumps(item, ensure_ascii=False))


    with open(input_path, "w", encoding="utf-8") as f:
        f.write("\n".join(modified_lines) + "\n")

    print(f"\n✅ All noise signs removed: {noise_pattern.pattern}")
    print(f"📊 Total lines cleaned: {cleaned_count}")

def clean_curly_braces(file_path):

    def remove_braces_and_track_deletions(text):
        words = text.split()
        new_words = []
        deleted_indices = []
        i = 0

        while i < len(words):
            word = words[i]
            if re.match(r"^\{[^ ]+\}$", word):

                deleted_indices.append(i)
            elif "{" in word and "}" in word:

                new_word = re.sub(r"\{([^ ]*?)\}", r"\1", word)
                new_words.append(new_word)
            elif "{" in word:

                start_idx = i
                while i < len(words) and "}" not in words[i]:
                    deleted_indices.append(i)
                    i += 1
                if i < len(words) and "}" in words[i]:
                    deleted_indices.append(i)

            else:
                new_words.append(word)
            i += 1

        return " ".join(new_words), deleted_indices

    modified_count = 0
    cleaned_data = []
    modified_sentences = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            data = json.loads(line.strip())

            original_text, deleted_indices = remove_braces_and_track_deletions(data.get("original_text", ""))
            lemmatized_text, deleted_indices_lemmatized = remove_braces_and_track_deletions(
                data.get("lemmatized_text", ""))


            all_deleted_indices = set(deleted_indices + deleted_indices_lemmatized)

            upos = data.get("UPOS", "").split()
            xpos = data.get("XPOS", "").split()
            feature = data.get("feature", "").split()


            upos = [upos[i] for i in range(len(upos)) if i not in all_deleted_indices]
            xpos = [xpos[i] for i in range(len(xpos)) if i not in all_deleted_indices]
            feature = [feature[i] for i in range(len(feature)) if i not in all_deleted_indices]

            updated_data = data.copy()
            updated_data.update({
                "original_text": original_text,
                "lemmatized_text": lemmatized_text,
                "UPOS": " ".join(upos),
                "XPOS": " ".join(xpos),
                "feature": " ".join(feature)
            })

            if data != updated_data:
                modified_count += 1
                modified_sentences.append((data["original_text"], original_text))

            file.write(json.dumps(updated_data, ensure_ascii=False) + "\n")

    print(f"revised {modified_count} JSONL sentences")
    for before, after in modified_sentences:
        print(f"before: {before}\nafter: {after}\n")

def remove_colon_like_symbols(text):
    words = text.split()
    new_words = []
    deleted_indices = []

    for i, word in enumerate(words):
        if word in {":", ":—", "⁛—", "⁘ⲻ", "⁖ⲻ", "✠"}:
            deleted_indices.append(i)
        elif any(sym in word for sym in {":—", ":", "⁛—", "⁘ⲻ", "⁖ⲻ", "✠"}):

            new_word = re.sub(r":—|:|⁛—|⁘ⲻ|⁖ⲻ|✠", "", word)
            new_words.append(new_word)
        else:
            new_words.append(word)

    return " ".join(new_words), deleted_indices

def clean_colons(file_path):

    modified_count = 0
    modified_sentences = []

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            data = json.loads(line.strip())


            original_text, deleted_indices_colon = remove_colon_like_symbols(data.get("original_text", ""))
            lemmatized_text, deleted_indices_colon_lemmatized = remove_colon_like_symbols(
                data.get("lemmatized_text", ""))

            all_deleted_indices = set(deleted_indices_colon + deleted_indices_colon_lemmatized)

            upos = data.get("UPOS", "").split()
            xpos = data.get("XPOS", "").split()
            feature = data.get("feature", "").split()


            upos = [upos[i] for i in range(len(upos)) if i not in all_deleted_indices]
            xpos = [xpos[i] for i in range(len(xpos)) if i not in all_deleted_indices]
            feature = [feature[i] for i in range(len(feature)) if i not in all_deleted_indices]


            updated_data = data.copy()
            updated_data.update({
                "original_text": original_text,
                "lemmatized_text": lemmatized_text,
                "UPOS": " ".join(upos),
                "XPOS": " ".join(xpos),
                "feature": " ".join(feature)
            })

            if data != updated_data:
                modified_count += 1
                modified_sentences.append((data["original_text"], original_text))

            file.write(json.dumps(updated_data, ensure_ascii=False) + "\n")

    print(f"revised {modified_count} sentences")
    for before, after in modified_sentences:
        print(f"before: {before}\nafter: {after}\n")

def replace_slashes(file_path):

    modified_count = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for line in lines:
            data = json.loads(line.strip())

            original_text = data.get("original_text", "")
            lemmatized_text = data.get("lemmatized_text", "")
            upos = data.get("UPOS", "")
            xpos = data.get("XPOS", "")
            feature = data.get("feature", "")

            words = original_text.split()
            lemmatized_words = lemmatized_text.split()
            upos_list = upos.split()
            xpos_list = xpos.split()
            feature_list = feature.split()

            new_words = []
            new_lemmatized_words = []
            new_upos = []
            new_xpos = []
            new_feature = []

            for i, word in enumerate(words):
                if word == "ⲧⲁ\\'ⲩⲭⲏ" or word == "ⲧⲁ\'ⲩⲭⲏ":
                    new_words.extend(["ⲧⲁ", "ⲯⲩⲭⲏ"])
                    new_lemmatized_words.extend(["ⲡⲁ", "ⲯⲩⲭⲏ"])
                    new_upos.extend(["DET", "NOUN"])
                    new_xpos.extend(["PPOS", "N"])
                    new_feature.extend(
                        ["Definite=Def|Gender=Fem|Number=Sing|Number[psor]=Sing|Person=1|Poss=Yes|PronType=Prs",
                         "Foreign=Yes"])
                else:
                    new_words.append(word)
                    new_lemmatized_words.append(lemmatized_words[i])
                    new_upos.append(upos_list[i])
                    new_xpos.append(xpos_list[i])
                    new_feature.append(feature_list[i])

            updated_data = data.copy()
            updated_data.update({
                "original_text": " ".join(new_words),
                "lemmatized_text": " ".join(new_lemmatized_words),
                "UPOS": " ".join(new_upos),
                "XPOS": " ".join(new_xpos),
                "feature": " ".join(new_feature)
            })

            if data != updated_data:
                modified_count += 1

            file.write(json.dumps(updated_data, ensure_ascii=False) + "\n")

    print(f"revised {modified_count} JSONL sentences")





def search_non_special_chars(jsonl_path):

    count = 0
    modified_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            original_text = item.get("original_text", "")


            invalid_chars = []
            invalid_words = set()

            for match in non_special_char_pattern.finditer(original_text):
                invalid_char = match.group()
                invalid_chars.append(invalid_char)


                words = original_text.split()
                for word in words:
                    if invalid_char in word:
                        invalid_words.add(word)
                        break

            if invalid_chars:
                count += 1
                item["clean"] = "special signs"


                print(f"⚠️  illegal signs: {', '.join(invalid_chars)}")
                print(f"   in: {', '.join(invalid_words)}")
                print(f"   full text: {original_text}\n")

            modified_data.append(item)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in modified_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print(f"⚠️  discovered {count} lines including special signs，updated `{jsonl_path}` revising `clean` ！")

def search_special_chars(jsonl_path, output_path="review_examples_special.jsonl"):

    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in f:
            item = json.loads(line.strip())
            original_text = item.get("original_text", "")

            if special_char_pattern.search(original_text):
                count += 1
                json.dump(item, out_f, ensure_ascii=False)
                out_f.write("\n")  # 确保换行

    print(f"⚠️  总共发现 {count} 行包含 special_char_pattern 里的字符，并已保存至 `{output_path}`！")


check_missing_fields("manually_cleaned_cop_sah.jsonl")
check_missing_fields("manually_cleaned_cop_boh.jsonl")


remove_duplicates("manually_cleaned_cop_sah.jsonl")
remove_duplicates("manually_cleaned_cop_boh.jsonl") 


replace_gaps_and_update_clean("manually_cleaned_cop_sah.jsonl", "manually_cleaned_cop_sah_gap_fixed.jsonl")
replace_gaps_and_update_clean("manually_cleaned_cop_boh.jsonl", "manually_cleaned_cop_boh_gap_fixed.jsonl")




replace_gap_dots_asterisk_unknown_words("manually_cleaned_cop_sah_gap_fixed.jsonl")
replace_gap_dots_asterisk_unknown_words("manually_cleaned_cop_boh_gap_fixed.jsonl")




remove_noise_signs("manually_cleaned_cop_sah_gap_fixed.jsonl")
remove_noise_signs("manually_cleaned_cop_boh_gap_fixed.jsonl")


clean_curly_braces("manually_cleaned_cop_sah_gap_fixed.jsonl")
clean_curly_braces("manually_cleaned_cop_boh_gap_fixed.jsonl")



clean_colons("manually_cleaned_cop_sah_gap_fixed.jsonl")
clean_colons("manually_cleaned_cop_boh_gap_fixed.jsonl")


non_special_char_pattern = re.compile(r"[^a-zA-ZⲀ-⳿Ϣ-ϯͰ-Ͽἀ-῝-ͯ̄ …\[\].·ϧ\‐']")
special_char_pattern = re.compile(r"[{}]")  # 这里可以根据宝宝的需求增加更多不允许的字符


# 暂时可能不需要看这部分
search_non_special_chars("manually_cleaned_cop_sah_gap_fixed.jsonl")
search_non_special_chars("manually_cleaned_cop_boh_gap_fixed.jsonl")
