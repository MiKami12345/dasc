from flask import Flask, render_template, request, redirect, url_for
import csv

app = Flask(__name__)



## 色覚検査

# 最初のセットの問題数
num_tests_first_set = 3
# 二番目のセットの問題数
num_tests_second_set = 3

# 最初のセットの問題と正解
first_set_color_tests = [
    {'question': '赤', 'options': ['赤', '青', '黄'], 'answer': '赤', 'display_answer': '赤'},
    {'question': '青', 'options': ['赤', '青', '黄'], 'answer': '青', 'display_answer': '青'},
    {'question': '黄', 'options': ['赤', '青', '黄'], 'answer': '黄', 'display_answer': '黄'},
]

# 二番目のセットの問題と正解
second_set_color_tests = [
    {'question': '紫', 'options': ['赤', '青', '黄'], 'answer': '青', 'display_answer': '紫'},
    {'question': 'オレンジ', 'options': ['赤', '青', '黄'], 'answer': '黄', 'display_answer': 'オレンジ'},
    {'question': 'ピンク', 'options': ['赤', '青', '黄'], 'answer': '赤', 'display_answer': 'ピンク'},
]

# 現在の問題インデックス
current_question_index = 0

# 正解数を格納するリスト
correct_answers_list = [0] * (num_tests_first_set + num_tests_second_set)

# ホームページ
@app.route('/color')
def home():
    global current_question_index, correct_answers_list
    current_question_index = 0
    correct_answers_list = [0] * (num_tests_first_set + num_tests_second_set)
    return render_template('color/home.html')

# 問題を開始するページ
@app.route('/color/index')
def index():
    global current_question_index, correct_answers_list
    current_question_index = 0
    correct_answers_list = [0] * (num_tests_first_set + num_tests_second_set)
    return render_template('color/index.html', question=first_set_color_tests[current_question_index])

# チェックボタンを押した際の処理
@app.route('/color/check_answer', methods=['POST'])
def check_answer():
    global current_question_index, correct_answers_list
    user_answer = request.form['user_answer']

    # インデックスが範囲内かどうかを確認
    if current_question_index < num_tests_first_set + num_tests_second_set:
        if current_question_index < num_tests_first_set:
            current_question = first_set_color_tests[current_question_index]
        else:
            current_question = second_set_color_tests[current_question_index - num_tests_first_set]

        # ユーザーが選択した回答を比較
        # 表示は赤、青、黄だが、内部的には紫、青、オレンジとして処理
        if user_answer == current_question['answer']:
            correct_answers_list[current_question_index] = 1

        current_question_index += 1

    if current_question_index < num_tests_first_set + num_tests_second_set:
        if current_question_index < num_tests_first_set:
            current_question = first_set_color_tests[current_question_index]
        else:
            current_question = second_set_color_tests[current_question_index - num_tests_first_set]

        return render_template('color/index.html', question=current_question)
    else:
        return redirect(url_for('result'))

# 結果画面
@app.route('/color/result')
def result():
    global correct_answers_list
    correct_answers = sum(correct_answers_list)
    # current_question_index = 0  # リセットしない
    return render_template('color/result.html', correct_answers=correct_answers, num_tests=(num_tests_first_set + num_tests_second_set))

## 知識問題

# Load questions from CSV file
def load_questions():
    questions = []
    with open('static/csv/driving_questions.csv', 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        # print(csv_reader)
        for row in csv_reader:
            # print(row)
            # options = row['options'].split(',')
            question = {
                'question': row['\ufeff問題文'],
                'options': ["〇","×"],
                'correct_option': row['正解']
            }
            questions.append(question)
    return questions

questions = load_questions()

@app.route('/questions')
def driving_exam():
    return render_template('questions/driving_exam.html', questions=questions)

@app.route('/questions/check_exam', methods=['POST'])
def check_exam():
    score = 0
    for i, question in enumerate(questions, 1):
        user_answer = int(request.form[f'question{i}'])
        if user_answer == question['correct_option']:
            score += 1

    result_message = f'正解数: {score}/{len(questions)}'
    return result_message

## 反射神経

@app.route('/reflexes')
def reflexesIndex():
    return render_template('reflexes/index.html')


if __name__ == '__main__':
    app.run(debug=True)
