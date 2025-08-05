from collections import Counter
import numpy as np
import pandas as pd
import os

class Game():

    def __init__(self, size: int = 4, level: int = 'normal', language: str = 'russian', first_move: str = 'player'):

        assert level in {'hard', 'normal', 'easy'}, f'levels are "easy", "normal", "hard"'
        self.level = ['easy', 'normal', 'hard'].index(level)
        assert language.lower() in {'russian', 'german', 'english', 'french'}
        assert first_move.lower() in {'player', 'computer'}
        self.language = language.lower()
        self.first_move = first_move.lower()
        
        self.size = size
        self.field = [[' ' for _ in range(size)] for _ in range(size)]

        current_dir = os.path.dirname(__file__)
        self.csv_path = os.path.join(current_dir, f"{language.lower()}.csv")

        all_nouns = pd.read_csv(self.csv_path, sep='\t')['Lemma'].to_list()
        frequencies = pd.read_csv(self.csv_path, sep='\t')[['Lemma', 'Freq']]
        frequencies = frequencies.drop_duplicates(subset='Lemma')
        frequencies.set_index('Lemma', inplace=True)
        frequencies = frequencies.T.to_dict()
        self.frequencies = {key: value['Freq'] for key, value in frequencies.items()}
        # all_nouns = open('nouns.txt', 'r').read().split('\n')
        all_nouns = list(filter(lambda x: isinstance(x, str) and not '-' in x and not x[0] == x[0].upper() and len(x) > 2, all_nouns))

        if self.language == 'russian':
            self.alphabet = [chr(i) for i in list(range(4*256 + 3*16 + 0, 4*256 + 3*16 + 0 + 32))]
            self.single_letter_words = ['а', 'в', 'и', 'к', 'о', 'с', 'у', 'я']
            self.two_letter_words = ['ад', 'аз', 'ар', 'ас', 'го', 'до', 'ёж', 'ил', 'кю', 'ля', 'ме', 'ми', 'мо', 'ни', 'ню', 'ов', 'ом', 'он', 'ор', 'па', 'пе', 'пи', 'ре', 'си', 'су', 'то', 'уд', 'уж', 'ум', 'ус', 'ут', 'фа', 'ча', 'ши', 'шу', 'щи', 'юг', 'юз', 'юр', 'юс', 'ют', 'яд', 'як', 'ял', 'ям', 'яр', 'ад', 'еж', 'уж', 'щи', 'юг', 'яд', 'як', 'ям']
        elif self.language == 'english':
            self.alphabet = [chr(i) for i in list(range(97, 97+26))]
            self.single_letter_words = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x', 'y']
            self.two_letter_words = ["an", "am", "as", "at", "be", "by", "do", "ex", "go", "he", "hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"]
    
        self.all_nouns = self.single_letter_words + all_nouns + self.two_letter_words
        self.max_noun_length = max([len(noun) for noun in self.all_nouns])

        self.candidates = set()
        self.n_grams = {i: set() for i in range(2, self.max_noun_length)}

        length_groups = {i: set() for i in range(1, self.max_noun_length + 1)}
        for noun in self.all_nouns:
            length_groups[len(noun)].add(noun)
        self.length_groups = length_groups

        # print(self.length_groups[1])

        letter_groups = {}
        for length, sub_nouns in self.length_groups.items():
            letter_groups.update({length: {}})
            for noun in sub_nouns:
                if any(not l.isalpha() for l in noun):
                    continue
                unique_letters = Counter(noun)
                for letter, value in unique_letters.items():
                    for i in range(1, value+1):
                        key = letter * i
                        if key not in letter_groups[length]:
                            letter_groups[length].update({key: set()})
                        letter_groups[length][key].add(noun)

        self.letter_groups = letter_groups

        self.filled_places = []
        self.used_letters = []
        self.empty_places = [(i, j) for i in range(self.size) for j in range(self.size)]

        self.computer_score = 0
        self.player_score = 0

        self.computer_logs = []
        self.player_logs = []

        if self.first_move == 'computer':
            self.computer_move()

    def get_ngrams(self, word):

        # n_grams = {i: set() for i in range(2, self.max_noun_length)}
        if len(word) >= 2:
            for n in range(2, len(word) + 1):
                self.n_grams[n].add(word[:n])
        
        return

    def possible_places(self):
        if len(self.filled_places) == 0:
            return self.empty_places.copy()
        else:
            pps = []
            for (i, j) in self.empty_places:
                if (i+1, j) in self.filled_places or (i-1, j) in self.filled_places or (i, j+1) in self.filled_places or (i, j-1) in self.filled_places:
                    pps.append((i, j))
            return pps
        
    def get_all_paths(self, new_letter, new_place, length, search_word: str = None):

        if length > self.max_noun_length:
            return set()

        new_field = [[ff + '' for ff in f] for f in self.field]
        new_field[new_place[0]][new_place[1]] = new_letter

        new_filled = self.filled_places.copy()
        new_filled.append(new_place)

        current_length = 0
        paths = []
        while current_length < length:

            if current_length == 0:
                paths += [[(p[0], p[1], new_field[p[0]][p[1]])] for p in new_filled]
                current_length += 1

            else:
                paths_right = [p + [(p[-1][0], p[-1][1] + 1, new_field[p[-1][0]][p[-1][1] + 1])] for p in paths 
                               if (p[-1][1] + 1 < self.size and 
                                   (p[-1][0], p[-1][1] + 1) in new_filled and 
                                   (p[-1][0], p[-1][1] + 1, new_field[p[-1][0]][p[-1][1] + 1]) not in p)]
                
                paths_left = [p + [(p[-1][0], p[-1][1] - 1, new_field[p[-1][0]][p[-1][1] - 1])] for p in paths 
                              if (p[-1][1] - 1 >= 0 and 
                                  (p[-1][0], p[-1][1] - 1) in new_filled and 
                                  (p[-1][0], p[-1][1] - 1, new_field[p[-1][0]][p[-1][1] - 1]) not in p)]
                
                paths_up = [p + [(p[-1][0] - 1, p[-1][1], new_field[p[-1][0] - 1][p[-1][1]])] for p in paths 
                            if (p[-1][0] - 1 >= 0 and 
                                (p[-1][0] - 1, p[-1][1]) in new_filled and 
                                (p[-1][0] - 1, p[-1][1], new_field[p[-1][0] - 1][p[-1][1]]) not in p)]
                
                paths_down = [p + [(p[-1][0] + 1, p[-1][1], new_field[p[-1][0] + 1][p[-1][1]])] for p in paths 
                              if (p[-1][0] + 1 < self.size and 
                                  (p[-1][0] + 1, p[-1][1]) in new_filled and 
                                  (p[-1][0] + 1, p[-1][1], new_field[p[-1][0] + 1][p[-1][1]]) not in p)]
                
                paths = paths_right + paths_left + paths_up + paths_down

                if current_length >= 2:
                    
                    if not search_word:
                        n_grams = self.n_grams[current_length + 1]
                    else:
                        n_grams = {search_word[:current_length+1]}
                    # print(f'Current length: {current_length + 1}, respective n_grams: {n_grams}')

                    paths_ = []
                    for p in paths:
                        # print(p)
                        if ''.join([pp[-1] for pp in p]) in n_grams:
                            paths_.append(p)
                    paths = paths_[:]
                
                current_length += 1

        paths = list(filter(lambda x: (*new_place, new_letter) in x, paths))
        return paths
    
    def get_candidates(self):

        options_groups = {i: set() for i in range(1, len(self.filled_places) + 2)}
        options = set()

        used_letters_keys = set()
        for key, value in Counter(self.used_letters).items():
            for i in range(1, value + 1):
                used_letters_keys.add(key * i)

        # print(f'[ Computer ] used letters as keys: {used_letters_keys}')

        for length in range(1, min(len(self.filled_places) + 2, 21)):
            
            # print(f'[ Computer ] searching for length {length}')

            for letter in self.alphabet:

                used_letters_keys_with_letter = used_letters_keys | {key + letter for key in used_letters_keys if key[0] == letter} | {letter}

                # print(f'[ Computer ] used letters as keys + {letter}: {used_letters_keys_with_letter}')

                length_group = self.letter_groups[length]

                exclude = set()
                for key, set_value in length_group.items():

                    if key not in used_letters_keys_with_letter:

                        exclude |= set_value

                include = self.length_groups[length] - exclude

                # print(f'[ Computer ] included in candidates: {include}')
                options |= {(inc, letter) for inc in include if letter in inc}
                options_groups[length] |= {(inc, letter) for inc in include if letter in inc}
                    
        # print(f'[ Computer ] candidates groups: {options_groups}')
    
        return options, options_groups 

    def find_best_word(self):

        possible_places = self.possible_places()
        options, options_groups = self.get_candidates()
        self.candidates = {o[0] for o in options}
        possible_letters = {option[1] for option in options}

        self.n_grams = {i: set() for i in range(2, self.max_noun_length)}
        for candidate in self.candidates:
            self.get_ngrams(candidate)

        results = set()
        lengths_found = 0

        for length in range(len(self.filled_places) + 1, 0, -1):
            
            done = False
            
            if lengths_found == 3 - self.level:
                break

            candidates = options_groups[length]
            possible_letters = {op[1] for op in options_groups[length]}

            for letter in possible_letters:

                for place in possible_places:

                    paths = self.get_all_paths(new_letter=letter, new_place=place, length=length)
                    word_paths = {(''.join(w[-1] for w in ww), letter) for ww in paths}

                    found = candidates & word_paths
                    found_new = set()
                    for f in found:
                        if f[0] in self.computer_logs or f[0] in self.player_logs:
                            continue
                        found_new.add(f)
                    if found_new:
                        done = True
                        for f in found_new:
                            results.add((f[0], f[1], place))

            if done:
                lengths_found += 1

        return results
    
    def put_letter(self, letter, place):

        letter = letter.lower()

        self.field[place[0]][place[1]] = letter
        self.filled_places.append(place)
        self.used_letters.append(letter)
        self.empty_places = [p for p in self.empty_places if p not in self.filled_places]

    def __str__(self):

        string = '\033[4m' + '|'.join([' ', *[str(i) for i in range(self.size)]]) + '\033[0m' + '\n' 
        # string += '_' * (self.size + 1) * 2 + '\n'

        for i, line in enumerate(self.field):
            
            string += str(i) + '|' + ' '.join([s if s != ' ' else '.' for s in line]) + '\n'

        string += 'Computer moves: '.ljust(20) + str(self.computer_logs) + '\n'
        string += 'Player moves: '.ljust(20) + str(self.player_logs) + '\n'
        string += 'Computer: '.ljust(20) + str(self.computer_score) + '\n'
        string += 'Player: '.ljust(20) + str(self.player_score) + '\n'
        return string 
    
    def computer_move(self):

        if len(self.empty_places) == 0:
            print('Game is finished!')
            return None, None, None, []

        if len(self.filled_places) == 0:
            p = np.random.randint(low=0, high=self.size ** 2)
            place = (p // self.size, p % self.size)
            letter = str(np.random.choice(self.single_letter_words))
            print(f'[ Computer ]: Letter: {letter} in place {place} -> {letter} (1)\n')
            word = letter

            path_used = [[place[0], place[1], letter]]

        else:
            
            words = self.find_best_word()
            words = sorted(words, key=lambda x: self.frequencies[x[0]] if x[0] in self.frequencies else 0, reverse=True)
            print(words)

            max_length = max([len(w[0]) for w in words])
            if max_length > 1:
                words = list(filter(lambda x: len(x[0]) > 1, words))
            print(f'[ Computer ] suggestions: {words}')
            prob = np.array([0.8 ** j for j in range(len(words))])
            prob = prob / np.sum(prob)
            i = np.random.choice(np.arange(len(words)), p=prob)
            word = words[i]
            word, letter, place = word
            print(f'[ Computer ]: Letter: {letter} in place {place} -> {word} ({len(word)})\n')

            paths = self.get_all_paths(new_letter=letter, new_place=place, length=len(word))
            path_used = next((p for p in paths if ''.join([pp[-1] for pp in p]) == word), None)
        
        self.computer_score += len(word)
        self.put_letter(letter, place)
        self.computer_logs.append(word)

        if len(self.empty_places) == 0:
            print('Game is finished!')
            return None, None, None, []

        print(word, letter, place, path_used)
        return word, letter, place, path_used

    def player_move(self, letter, place, word, ignore_check: bool = False):

        assert isinstance(letter, str), 'letter must be str'
        assert len(letter) == 1, 'letter must have 1 symbol'
        assert isinstance(place, tuple) and len(place) == 2, 'place must be 2-tuple'
        assert 0 <= place[0] <= self.size - 1 and 0 <= place[1] <= self.size - 1, f'place indices out of range'
        assert place in self.empty_places, f'this place is already occupied'
        assert isinstance(word, str) and len(word) >= 1, f'word must be a non-empty string'

        letter = letter.lower()
        word = word.lower()

        if len(self.empty_places) == 0:
            print('Game is finished!')
            return True

        if len(self.filled_places) == 0:
            
            assert letter in self.single_letter_words, f'no such word {letter}'

            print(f'[ Player ]: Letter: {letter} in place {place} -> {word} (1)\n')

        else:
            
            assert place in self.possible_places(), f'impossible place {place}'
            if not ignore_check:
                assert word in self.all_nouns, f'unknown word {word}'
            else:
                if word not in self.all_nouns:
                    pd.DataFrame({'Lemma': [word], 'Freq': [0.0]}, index=[0]).to_csv(self.csv_path, mode='a', index=False, sep='\t', header=False)

            paths = self.get_all_paths(letter, place, len(word), search_word=word)
            word_paths = {''.join(w[-1] for w in ww) for ww in paths}

            assert word in word_paths, f'impossible to construct the word {word}'
            
            print(f'[ Player ]: Letter: {letter} in place {place} -> {word} ({len(word)})\n')
        
        self.player_score += len(word)
        self.put_letter(letter, place)
        self.player_logs.append(word)

        if len(self.empty_places) == 0:
            print('Game is finished!')
            return True
        
        return False
    
    def autoplay(self):

        game = Game.load_game('\n'.join([''.join(['.' if not symbol.isalpha() else symbol for symbol in line]) for line in self.field]), 
        level=['easy', 'normal', 'hard'][self.level], language=self.language)

        done = False
        while True:

            done = game.computer_move()
            print(game)
            
            if done is True:
                break

        print('Game is over!')

    @staticmethod
    def load_game(game: str, **kwargs):

        game = list(filter(lambda x: x, game.replace(' ', '').split('\n')))
        size = len(game)
        game = [list(g) for g in game]
        filled = []
        for i, g in enumerate(game):
            assert len(g) == size, f'game must be a square field'
            for j, gg in enumerate(g):
                if gg.isalpha():
                    filled.append((gg, (i, j)))

        new_game = Game(size, **kwargs)
        for letter, place in filled:
            new_game.put_letter(letter, place)

        return new_game
    
    def play(self):

        done = False
        while True:

            done = self.computer_move()
            print(self)
            if done is True:
                break
            
            while True:

                try:
                    inp = input(f'Enter the letter, row, column and word, (ignore_check) separated by commas:')
                    if len(inp.split(',')) == 4:
                        letter, row, column, word = inp.replace(' ', '').split(',')
                        ignore_check = False
                    else:
                        letter, row, column, word, ignore_check = inp.replace(' ', '').split(',')
                    row, column = int(row), int(column)
                    done = self.player_move(letter, (row, column), word, ignore_check)
                    print(self)
                    break
                except BaseException as e:
                    print(f'Error: {e}, try once more!')

            if done:
                break
        print('Game is over!')
        if self.computer_score > self.player_score:
            print(f'Computer won! {self.computer_score}:{self.player_score}')
        elif self.computer_score < self.player_score:
            print(f'Player won! {self.computer_score}:{self.player_score}')
        else:
            print(f'Draw! {self.computer_score}:{self.player_score}')
