# NAMES = [ ## One token only, from https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py
#     "Michael",
#     "Christopher",
#     "Jessica",
#     "Matthew",
#     "Jennifer",
#     "Joshua",
#     "Amanda",
#     "Daniel",
#     "David",
#     "James",
#     "Robert",
#     "John",
#     "Joseph",
#     "Andrew",
#     "Ryan",
#     "Jason",
#     "Justin",
#     "Sarah",
#     "William",
#     "Jonathan",
#     "Brian",
#     "Nicole",
#     "Nicholas",
#     "Anthony",
#     "Eric",
#     "Elizabeth",
#     "Adam",
#     "Kevin",
#     "Steven",
#     "Thomas",
#     "Kyle",
#     "Rachel",
#     "Laura",
#     "Lauren",
#     "Richard",
#     "Amy",
#     "Michelle",
#     "Jeremy",
#     "Benjamin",
#     "Mark",
#     "Emily",
#     "Aaron",
#     "Charles",
#     "Rebecca",
#     "Jacob",
#     "Stephen",
#     "Patrick",
#     "Sean",
#     "Jamie",
#     "Kelly",
#     "Nathan",
#     "Sara",
#     "Paul",
#     "Angela",
#     "Tyler",
#     "Scott",
#     "Andrea",
#     "Gregory",
#     "Mary",
#     "Lisa",
#     "Bryan",
#     "Jose",
#     "Alexander",
#     "Jesse",
#     "Samuel",
# ]
ABCD_TEMPLATES = [
    "Then, [A] and [B] went to the [PLACE]. [C] gave a [OBJECT] to [D]",
    "Then, [A] and [B] had a lot of fun at the [PLACE]. [C] gave a [OBJECT] to [D]",
    "Then, [A] and [B] were working at the [PLACE]. [C] decided to give a [OBJECT] to [D]",
    "Then, [A] and [B] were thinking about going to the [PLACE]. [C] wanted to give a [OBJECT] to [D]",
    "Then, [A] and [B] had a long argument, and afterwards [C] said to [D]",
    "After [A] and [B] went to the [PLACE], [C] gave a [OBJECT] to [D]",
    "When [A] and [B] got a [OBJECT] at the [PLACE], [C] decided to give it to [D]",
    "When [A] and [B] got a [OBJECT] at the [PLACE], [C] decided to give the [OBJECT] to [D]",
    "While [A] and [B] were working at the [PLACE], [C] gave a [OBJECT] to [D]",
    "While [A] and [B] were commuting to the [PLACE], [C] gave a [OBJECT] to [D]",
    "After the lunch, [A] and [B] went to the [PLACE]. [C] gave a [OBJECT] to [D]",
    "Afterwards, [A] and [B] went to the [PLACE]. [C] gave a [OBJECT] to [D]",
    "Then, [A] and [B] had a long argument. Afterwards [C] said to [D]",
    "The [PLACE] [A] and [B] went to had a [OBJECT]. [C] gave it to [D]",
    "Friends [A] and [B] found a [OBJECT] at the [PLACE]. [C] gave it to [D]",
]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

# NAMES_2 = [ # Popular names for births in 1925-2024, https://www.ssa.gov/oact/babynames/decades/century.html
# "James"
# "Michael"
# "John"
# "Robert"
# "David"
# "William"
# "Richard"
# "Joseph"
# "Thomas"
# "Christopher"
# "Charles"
# "Daniel"
# "Matthew"
# "Anthony"
# "Mark"
# "Steven"
# "Donald"
# "Andrew"
# "Joshua"
# "Paul"
# "Kenneth"
# "Kevin"
# "Brian"
# "Timothy"
# "Ronald"
# "Jason"
# "George"
# "Edward"
# "Jeffrey"
# "Ryan"
# "Jacob"
# "Nicholas"
# "Gary"
# "Eric"
# "Jonathan"
# "Stephen"
# "Larry"
# "Justin"
# "Benjamin"
# "Scott"
# "Brandon"
# "Samuel"
# "Gregory"
# "Alexander"
# "Patrick"
# "Frank"
# "Jack"
# "Raymond"
# "Dennis"
# "Tyler"
# "Aaron"
# "Jerry"
# "Jose",
# "Nathan",
# "Adam",
# "Henry",
# "Zachary",
# "Douglas",
# "Peter",
# "Noah",
# "Kyle",
# "Ethan",
# "Christian",
# "Jeremy",
# "Keith",
# "Austin",
# "Sean",
# "Roger",
# "Terry",
# "Walter",
# "Dylan",
# "Gerald",
# "Carl",
# "Jordan",
# "Bryan",
# "Gabriel",
# "Jesse",
# "Harold",
# "Lawrence",
# "Logan",
# "Arthur",
# "Bruce",
# "Billy",
# "Elijah",
# "Joe",
# "Alan",
# "Juan",
# "Liam",
# "Willie",
# "Mason",
# "Albert",
# "Randy",
# "Wayne",
# "Vincent",
# "Lucas",
# "Caleb",
# "Luke",
# "Bobby",
# "Isaac",
# "Bradley",
# "Mary",
# "Patricia",
# "Jennifer",
# "Linda",
# "Elizabeth",
# "Barbara",
# "Susan",
# "Jessica",
# "Karen",
# "Sarah",
# "Lisa",
# "Nancy",
# "Sandra",
# "Ashley",
# "Emily",
# "Kimberly",
# "Betty",
# "Margaret",
# "Donna",
# "Michelle",
# "Carol",
# "Amanda",
# "Melissa",
# "Deborah",
# "Stephanie",
# "Rebecca",
# "Sharon",
# "Laura",
# "Cynthia",
# "Amy",
# "Kathleen",
# "Angela",
# "Dorothy",
# "Shirley",
# "Emma",
# "Brenda",
# "Nicole",
# "Pamela",
# "Samantha",
# "Anna",
# "Katherine",
# "Christine",
# "Debra",
# "Rachel",
# "Olivia",
# "Carolyn",
# "Maria",
# "Janet",
# "Heather",
# "Diane",
# "Catherine",
# "Julie",
# "Victoria",
# "Helen",
# "Joyce",
# "Lauren",
# "Kelly",
# "Christina",
# "Joan",
# "Judith",
# "Ruth",
# "Hannah",
# "Evelyn",
# "Andrea",
# "Virginia",
# "Megan",
# "Cheryl",
# "Jacqueline",
# "Madison",
# "Sophia",
# "Abigail",
# "Teresa",
# "Isabella",
# "Sara",
# "Janice",
# "Martha",
# "Gloria",
# "Kathryn",
# "Ann",
# "Charlotte",
# "Judy",
# "Amber",
# "Julia",
# "Grace",
# "Denise",
# "Danielle",
# "Natalie",
# "Alice",
# "Marilyn",
# "Diana",
# "Beverly",
# "Jean",
# "Brittany",
# "Theresa",
# "Frances",
# "Kayla",
# "Alexis",
# "Tiffany",
# "Lori",
# "Kathy",
# ]

# 124 names, abridged from the name list of https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py and 
# Popular names for births in 1925-2024, https://www.ssa.gov/oact/babynames/decades/century.html
# we only use one-token names to simplify the discussion
NAMES = [
    "Joshua",
    "Barbara",
    "Rebecca",
    "David",
    "Samuel",
    "Harold",
    "Andrea",
    "Jennifer",
    "Christopher",
    "Gabriel",
    "Christian",
    "Mary",
    "Virginia",
    "Luke",
    "Gregory",
    "Catherine",
    "Nancy",
    "Lucas",
    "Lisa",
    "Julia",
    "Helen",
    "Julie",
    "Martha",
    "Kyle",
    "Benjamin",
    "Carol",
    "Adam",
    "Bobby",
    "Logan",
    "Dylan",
    "Laura",
    "Alan",
    "Stephen",
    "Scott",
    "Bryan",
    "James",
    "Mason",
    "Nicholas",
    "Kevin",
    "Jason",
    "Nicole",
    "Sean",
    "Grace",
    "Patrick",
    "Anna",
    "Matthew",
    "Juan",
    "Thomas",
    "Paul",
    "Lauren",
    "Alexander",
    "Austin",
    "Henry",
    "Elizabeth",
    "Bruce",
    "Eric",
    "Hannah",
    "Diana",
    "Keith",
    "Ann",
    "Jose",
    "Amanda",
    "Anthony",
    "Margaret",
    "Arthur",
    "Aaron",
    "Tyler",
    "Douglas",
    "Angela",
    "Jacob",
    "Wayne",
    "Justin",
    "Terry",
    "Michael",
    "Karen",
    "Jean",
    "Joe",
    "Amy",
    "Ruth",
    "Noah",
    "Frances",
    "Madison",
    "Steven",
    "Alice",
    "Sarah",
    "Jessica",
    "Linda",
    "Walter",
    "Kelly",
    "Rachel",
    "Vincent",
    "Isaac",
    "Sara",
    "Andrew",
    "Jesse",
    "Jordan",
    "Maria",
    "Robert",
    "Victoria",
    "Carl",
    "Jonathan",
    "Peter",
    "Mark",
    "Billy",
    "Jamie",
    "Daniel",
    "Nathan",
    "Charlotte",
    "William",
    "Lawrence",
    "Susan",
    "John",
    "Richard",
    "Ryan",
    "Joan",
    "Jeremy",
    "Albert",
    "Charles",
    "Emma",
    "Brian",
    "Joseph",
    "Michelle",
    "Emily",
    "Roger",
]

import copy
import random
random.seed(42) # for repeatability

def gen_ioi_prompt(N, tokenizer, prompt_type, nouns_dict, names, templates, prompt_dataset=None):
    """
    :param1 N: number of prompts
    :param2 tokenizer: the assigned tokenzier
    :param3 prompt_type: "ABBA", "ABAB", "mixed"(="ABBA"+"ABAB"), "loaded"(=assigned by param7: prompt_dataset)
    :param4 nouns_dict: {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}
    :param5 names: name list
    :param6 templates: template list
    :param7 prompt_dataset: provide a prompt_dataset constructed by this function, set 'None' if you don't have one
    :return: a list of dicts
    """
    prompt_counter = 0
    ioi_prompt_dict_ls = []
    while prompt_counter < N:
        if prompt_dataset is None:
            ioi_prompt_dict = {}
            # select a template
            cur_template = random.choice(templates)
            cur_template_id = templates.index(cur_template)
            ioi_prompt_dict["TEMPLATE_IDX"] = cur_template_id
            # select names
            name_1, name_2 = "", ""
            while name_1 == name_2:
                name_1 = random.choice(names)
                name_2 = random.choice(names)
            ioi_prompt_dict["IOI_A"] = name_1
            ioi_prompt_dict["IOI_B"] = name_2
            # select [PLACE] and [OBJECT]
            for k, v in nouns_dict.items():
                ioi_prompt_dict[k] = random.choice(v)
                # print(k, ioi_prompt_dict[k])
            # select a prompt type if it is undetermined
            if prompt_type == "mixed":
                ioi_prompt_dict["prompt_type"] = ("ABBA" if prompt_counter < (N // 2) else "ABAB")
            elif prompt_type in ["ABBA", "ABAB"]:
                ioi_prompt_dict["prompt_type"] = prompt_type
        else: # just load the existing prompt dataset
            ioi_prompt_dict = prompt_dataset[prompt_counter]
            cur_template = templates[ioi_prompt_dict["TEMPLATE_IDX"]]
        
        prompt = cur_template
        # fill [PLACE] and [OBJECT]
        for k in nouns_dict:
            prompt = prompt.replace(k, ioi_prompt_dict[k])
            # print(k, prompt)

        # fill the names
        prompt = prompt.replace("[A]", ioi_prompt_dict["IOI_A"])
        prompt = prompt.replace("[B]", ioi_prompt_dict["IOI_B"])

        
        if ioi_prompt_dict["prompt_type"] == "ABBA":
            prompt = prompt.replace("[C]", ioi_prompt_dict["IOI_B"])
            prompt = prompt.replace("[D]", ioi_prompt_dict["IOI_A"])
            ioi_prompt_dict["IO"] = ioi_prompt_dict["IOI_A"]
            ioi_prompt_dict["S"] = ioi_prompt_dict["IOI_B"]
        elif ioi_prompt_dict["prompt_type"] == "ABAB":
            prompt = prompt.replace("[C]", ioi_prompt_dict["IOI_A"])
            prompt = prompt.replace("[D]", ioi_prompt_dict["IOI_B"])
            ioi_prompt_dict["IO"] = ioi_prompt_dict["IOI_B"]
            ioi_prompt_dict["S"] = ioi_prompt_dict["IOI_A"]

        ioi_prompt_dict["text"] = prompt
        ioi_prompt_dict["tokens"] = tokenizer.encode(prompt)
        ioi_prompt_dict["IO_token_id"] = tokenizer.encode(" " + ioi_prompt_dict["IO"])
        ioi_prompt_dict["S_token_id"] = tokenizer.encode(" " + ioi_prompt_dict["S"])
        ioi_prompt_dict["IO_token_pos"] = ioi_prompt_dict["tokens"].index(ioi_prompt_dict["IO_token_id"][0])
        ioi_prompt_dict["S_token_pos"] = [i for i, e in enumerate(ioi_prompt_dict["tokens"]) if e == ioi_prompt_dict["S_token_id"][0]]
        ioi_prompt_dict["END_token_pos"] = ([i for i, e in enumerate(ioi_prompt_dict["tokens"]) if e == ioi_prompt_dict["IO_token_id"][0]][-1] - 1)
        ioi_prompt_dict["S1+1_token_pos"] = (ioi_prompt_dict["S_token_pos"][0] + len(ioi_prompt_dict["S_token_id"]))
        # NOTE: END_token_pos is the position of the preceding token of the last name in the sentence
        # NOTE: S_token_pos is a list: [S1_token_pos, S2_token_pos]
        # NOTE: IO_token_pos is the position of the first IO_token: [IO1_token_pos]
        ioi_prompt_dict_ls.append(ioi_prompt_dict)
        prompt_counter += 1

    return ioi_prompt_dict_ls

def gen_abc_prompt(tokenizer, templates, ioi_prompt_dataset, abc_prompt_dataset=None):
    new_prompt_dict_ls = copy.deepcopy(ioi_prompt_dataset)
    if abc_prompt_dataset is not None:
        abc_name_ls = [[i["ABC_A"], i["ABC_B"], i["ABC_C"]] for i in abc_prompt_dataset]
    prompt_counter = 0
    for new_prompt_dict in new_prompt_dict_ls:
        if abc_prompt_dataset is None:
            name_1, name_2, name_3 = "", "", ""
            while len(set([name_1, name_2, name_3, new_prompt_dict["IO"], new_prompt_dict["S"]])) < 5:
                name_1 = random.choice(NAMES)
                name_2 = random.choice(NAMES)
                name_3 = random.choice(NAMES)
                # print(name_1, name_2, name_3, new_prompt_dict["IO"], new_prompt_dict["S"])
        else:
            name_1, name_2, name_3 = abc_name_ls[prompt_counter]
        prompt = templates[new_prompt_dict["TEMPLATE_IDX"]]
        for k in ["[PLACE]", "[OBJECT]"]:
            prompt = prompt.replace(k, new_prompt_dict[k])
        prompt = prompt.replace("[A]", name_1)
        prompt = prompt.replace("[B]", name_2)
        prompt = prompt.replace("[C]", name_3)
        prompt = prompt.replace("[D]", new_prompt_dict["IO"])
        new_prompt_dict["text"] = prompt
        old_prompt_type = new_prompt_dict["prompt_type"]
        new_prompt_dict["prompt_type"] = "ABC"
        new_prompt_dict["tokens"] = tokenizer.encode(prompt)
        new_prompt_dict["ABC_A"] = name_1
        new_prompt_dict["ABC_B"] = name_2
        new_prompt_dict["ABC_C"] = name_3
        new_prompt_dict["A_token_id"] = tokenizer.encode(" " + new_prompt_dict["ABC_A"])
        new_prompt_dict["B_token_id"] = tokenizer.encode(" " + new_prompt_dict["ABC_B"])
        new_prompt_dict["C_token_id"] = tokenizer.encode(" " + new_prompt_dict["ABC_C"])
        new_prompt_dict["D_token_pos"] = new_prompt_dict["tokens"].index(new_prompt_dict["IO_token_id"][0])
        new_prompt_dict["A_token_pos"] = new_prompt_dict["tokens"].index(new_prompt_dict["A_token_id"][0])
        new_prompt_dict["B_token_pos"] = new_prompt_dict["tokens"].index(new_prompt_dict["B_token_id"][0])
        new_prompt_dict["C_token_pos"] = new_prompt_dict["tokens"].index(new_prompt_dict["C_token_id"][0])
        new_prompt_dict["END_token_pos"] = (new_prompt_dict["D_token_pos"] - 1)
        if old_prompt_type == "ABBA":
            new_prompt_dict["S1+1_token_pos"] = new_prompt_dict["B_token_pos"] + len(new_prompt_dict["B_token_id"])
        elif old_prompt_type == "ABAB":
            new_prompt_dict["S1+1_token_pos"] = new_prompt_dict["A_token_pos"] + len(new_prompt_dict["A_token_id"])
        prompt_counter += 1
    return new_prompt_dict_ls


####-------- Test code --------####
# from transformers import AutoTokenizer
# import pickle
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
# ioi_dataset_olmoe = gen_ioi_prompt(5000, tokenizer, "mixed", {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}, NAMES, ABCD_TEMPLATES, None)
# abc_dataset_olmoe = gen_abc_prompt(tokenizer, ABCD_TEMPLATES, ioi_dataset_olmoe)
# print(ioi_dataset_olmoe[1000])
# print(abc_dataset_olmoe[1000])

## an example for loading an existing dataset
# ioi_dataset2_olmoe = gen_ioi_prompt(5000, tokenizer, "mixed", {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}, NAMES, ABCD_TEMPLATES, ioi_dataset_olmoe)
# print(ioi_dataset2_olmoe[1000])
# abc_dataset2_olmoe = gen_abc_prompt(tokenizer, ABCD_TEMPLATES, ioi_dataset_olmoe, abc_dataset_olmoe)
# print(abc_dataset_olmoe[1000])

## save/load the dataset
# with open("ioi_dataset_olmoe.pkl", "wb") as f:
#     pickle.dump(ioi_dataset_olmoe, f)
# with open("ioi_dataset_olmoe.pkl", "rb") as f:
#     ioi_dataset_olmoe = pickle.load(f)

## TODO: we have not completely implemented the following functions yet, check them
## use the same dataset but on another model
# tokenizer2 = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
# ioi_dataset_mixtral = gen_ioi_prompt(5000, tokenizer2, "load", {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}, NAMES, ABCD_TEMPLATES, ioi_dataset_olmoe)
# abc_dataset_mixtral = gen_abc_prompt(tokenizer2, ABCD_TEMPLATES, ioi_dataset_mixtral, abc_dataset_olmoe) # TODO: token_pos has bug, fix it
# print(ioi_dataset_mixtral[1000])
# print(abc_dataset_mixtral[1000])

## check if the names corresponds to only one token to simplify the analysis (unused for now)
# temp_names = set(NAMES)
# counter = 0
# for i in temp_names:
#     if len(tokenizer2.encode(" "+i)) == 3 and len(tokenizer.encode(" "+i))==1:
#         print('"{}",'.format(i))
#         counter += 1
# print(counter)