import streamlit as st
from game import Game

# Default game settings
default_size = 4
default_level = 'normal'
default_language = 'English'

# Initialize settings
if "settings" not in st.session_state:
    st.session_state.settings = {"size": default_size, "level": default_level, 'language': default_language}

# Initialize game
if "game" not in st.session_state:
    st.session_state.game = Game(size=default_size, level=default_level)
    st.session_state.selected_cell = None
    st.session_state.letter = ""
    st.session_state.word = ""
    st.session_state.message = ""
    st.session_state.last_computer_path = set()
    st.session_state.error_word = None

if st.session_state.game.size == 3:
    side = 4
elif st.session_state.game.size == 6:
    side = 2
elif st.session_state.game.size == 5:
    side = 2
else:
    side = 3
# Button styling
custom_css = f"""
    <style>
    .scrollable-board {{
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 1em;
    }}
    .square-button {{
        height: {side}em;
        width: {side}em;
        font-size: 20px !important;
        background-color: #6666ee !important;
        padding: 0 !important;
        margin: 0.1em;
        text-align: center;
        line-height: {side}em;
        color: black;
    }}
    .highlight-cell {{
        background-color: #ffd966 !important;
        color: black !important;
        border: 2px solid #e69138 !important;
    }}
    .selected-cell {{
        height: {side-0.3}em;
        width: {side-0.3}em;
        background-color: #000000 !important;
        color: black !important;
        border: 2px solid red !important;
    }}
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

game = st.session_state.game
size = game.size

# Center alignment using container
st.markdown("""
    <style>
    .main > div {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Layout: 3 columns (settings - board - progress)
left_col, center_col, right_col = st.columns([1.5, 2.5, 1.5], gap="medium")

with left_col:
    st.markdown("## Game Settings")
    new_size = st.selectbox("Grid size", options=list(range(3, 7)), index=st.session_state.settings["size"] - 3)
    new_level = st.selectbox("Difficulty", options=["easy", "normal", "hard"], index=["easy", "normal", "hard"].index(st.session_state.settings["level"]))
    new_language = st.selectbox("Language", options=["English", "Russian"])
    new_first = st.selectbox("First Move", options=["Computer", "Player"])

    button_cols = st.columns(2)
    with button_cols[0]:
        if st.button("New Game"):
            st.session_state.settings = {
                "size": new_size,
                "level": new_level,
                "language": new_language,
                "first": new_first}
            st.session_state.game = Game(size=new_size, level=new_level, language=new_language, first_move=new_first)
            st.session_state.selected_cell = None
            st.session_state.letter = ""
            st.session_state.word = ""
            st.session_state.message = ""
            st.session_state.last_computer_path = set()
            st.session_state.error_word = None
            st.rerun()
    with button_cols[1]:
        if st.button("End Game"):
            st.session_state.message = "Game ended manually. Final scores shown."
            st.session_state.selected_cell = None

with center_col:
    st.title("Word Grid Game")
    st.caption("Click a valid cell, enter your letter and word, then submit.")
    st.markdown("### Game Board")

    st.markdown('<div class="scrollable-board">', unsafe_allow_html=True)

    possible = set(game.possible_places())
    for i in range(size):
        cols = st.columns(size, gap="small")
        for j in range(size):
            val = game.field[i][j]
            coord = (i, j)
            css_class = "square-button"
            if coord == st.session_state.selected_cell:
                css_class += " selected-cell"
            elif coord in st.session_state.last_computer_path:
                css_class += " highlight-cell"

            with cols[j]:
                if val != ' ':
                    st.markdown(f"""<button class='{css_class}' disabled>{val.upper()}</button>""", unsafe_allow_html=True)
                elif coord in possible:
                    if coord == st.session_state.selected_cell:
                        st.markdown(f"""<button class='{css_class}' disabled>{'&nbsp;' * 5}<br>{'&nbsp;' * 5}</button>""", unsafe_allow_html=True)
                    else:
                        if game.size == 3:
                            length = 6
                            rows = 5
                        elif game.size == 4:
                            length = 5
                            rows = 3
                        elif game.size == 5:
                            length = 4
                            rows = 1
                        elif game.size == 6:
                            length = 3
                            rows = 1
                        placeholder = '\n'.join(['\u00A0' * length for _ in range(rows)])
                        if st.button(placeholder, key=f"valid_{i}_{j}"):
                            st.session_state.selected_cell = coord
                            st.session_state.last_computer_path = set()
                            st.rerun()
                else:
                    st.markdown(f"""<button class='{css_class}' disabled>&nbsp;</button>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.selected_cell:
        st.info(f"Selected cell: {st.session_state.selected_cell}")
        st.session_state.letter = st.text_input("Your letter", value=st.session_state.letter, max_chars=1)
        st.session_state.word = st.text_input("Your word", value=st.session_state.word)

        if st.button("Submit Move"):
            try:
                done = game.player_move(
                    letter=st.session_state.letter.lower(),
                    place=st.session_state.selected_cell,
                    word=st.session_state.word.lower()
                )
                st.session_state.message = f"Move accepted: {st.session_state.word.lower()}"
                st.session_state.error_word = None

                st.session_state.selected_cell = None
                st.session_state.letter = ""
                st.session_state.word = ""

                if done:
                    if game.player_score > game.computer_score:
                        result = 'You won!'
                    elif game.player_score < game.computer_score:
                        result = 'You lost!'
                    else:
                        result = "It's a Draw!"

                    st.session_state.message += f"\n\n\n🎉 Game over: {result} \nFinal score — Player: {game.player_score}, Comp: {game.computer_score}"
                else:
                    last_word, last_letter, last_place, path_used = game.computer_move()
                    if last_word is None:
                        if game.player_score > game.computer_score:
                            result = 'You won!'
                        elif game.player_score < game.computer_score:
                            result = 'You lost!'
                        else:
                            result = "It's a Draw!"

                        st.session_state.message += f"\n\n\n🎉 Game over: {result} \nFinal score — Player: {game.player_score}, Comp: {game.computer_score}"
                        st.session_state.last_computer_path = set()
                    else:
                        st.session_state.last_computer_path = set((a, b) for a, b, _ in path_used)

                st.rerun()
            except Exception as e:
                st.session_state.message = f"Error: {str(e)}"
                if "unknown word" in str(e).lower():
                    st.session_state.error_word = {
                        "letter": st.session_state.letter.lower(),
                        "place": st.session_state.selected_cell,
                        "word": st.session_state.word.lower()
                    }

    if st.session_state.message:
        st.markdown(st.session_state.message)

    if st.session_state.error_word:
        if st.button("Add to Dictionary"):
            try:
                done = game.player_move(
                    letter=st.session_state.error_word["letter"],
                    place=st.session_state.error_word["place"],
                    word=st.session_state.error_word["word"],
                    ignore_check=True
                )
                st.session_state.message = f"Word '{st.session_state.error_word['word']}' added to dictionary."
                st.session_state.selected_cell = None
                st.session_state.letter = ""
                st.session_state.word = ""
                st.session_state.error_word = None

                if done:
                    if game.player_score > game.computer_score:
                        result = 'You won!'
                    elif game.player_score < game.computer_score:
                        result = 'You lost!'
                    else:
                        result = "It's a Draw!"

                    st.session_state.message += f"\n\n\n🎉 Game over: {result} \nFinal score — Player: {game.player_score}, Comp: {game.computer_score}"
                    st.session_state.last_computer_path = set()
                else:
                    last_word, last_letter, last_place, path_used = game.computer_move()
                    if last_word is None:

                        if game.player_score > game.computer_score:
                            result = 'You won!'
                        elif game.player_score < game.computer_score:
                            result = 'You lost!'
                        else:
                            result = "It's a Draw!"
                        st.session_state.message += f"\n\n\n🎉 Game over: {result} \nFinal score — Player: {game.player_score}, Comp: {game.computer_score}"
                        st.session_state.last_computer_path = set()
                    else:
                        st.session_state.last_computer_path = set((a, b) for a, b, _ in path_used)
                st.rerun()
            except Exception as e:
                st.session_state.message = f"Error while adding word: {str(e)}"

with right_col:
    st.markdown("## Game Progress")

    score_cols = st.columns(2)
    with score_cols[0]:
        st.markdown("### Comp")
        st.metric("Score", game.computer_score)
        for word in game.computer_logs:
            letter = word[0] if word else ""
            st.text(f"{letter}: {word}")

    with score_cols[1]:
        st.markdown("### Player")
        st.metric("Score", game.player_score)
        for word in game.player_logs:
            letter = word[0] if word else ""
            st.text(f"{letter}: {word}")
