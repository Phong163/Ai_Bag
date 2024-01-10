import json
from googletrans import Translator


def translator(input_text,lang):
    output=Translator().translate(input_text,dest=lang)
    return output.text
def classify_input(text):
    type1=['bật nhạc','kết thúc']
    type2=['dịch tiếng anh','dịch tiếng việt','dịch tiếng nhật','dịch tiếng trung quốc','dịch tiếng nga','dịch tiếng bồ đào nha','dịch tiếng pháp','dịch tiếng thái lan','dịch tiếng hàn quốc','dịch tiếng ấn độ']
    operators = ['cộng', 'trừ', 'nhân', 'chia', 'mũ', '+', '-', '*']
    if text in type1:
        type=1
    elif text in type2:
        type=2
    elif any(item in text for item in operators) and any(char.isdigit() for char in text):
        type=3
    else:
        type=4
    return type
def calculate(text):
    ds1 = []
    ds2 = []
    ds3 = []
    operators2 = {'cộng': '+', 'trừ': '-', 'nhân': '*', 'chia': '/', 'mũ': '**','+': '+', '-': '-', '*': '*'}
    for item in text:
        if item in operators2:
            ds1.append(operators2[item])
        elif item.isdigit():
            ds2.append(item)
    for i in range(len(ds1)):
        ds3.append(ds2[i])
        ds3.append(ds1[i])
    # Thêm phần tử cuối cùng của ds2 nếu cần
    if len(ds2) > len(ds1):
        ds3.append(ds2[-1])
    print(ds3)
    result_text = ''.join(ds3)
    print(result_text)
    result=eval(result_text)
    aw=f'kết quả là {result}'
    return aw
def command_type1(text):
    if text=='bật nhạc':
        answer='đây là bài hát tôi thường nghe'
        next_act=1
        return answer,next_act
    if text=='kết thúc':
        answer='tạm biệt'
        next_act=0
        return answer,next_act
def command_type2(text):
    if text=='dịch tiếng anh':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng anh'
        lang='en'
        return answer,lang
    if text=='dịch tiếng việt':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng việt'
        lang='vi'
        return answer,lang
    if text=='dịch tiếng nhật':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng nhật'
        lang='ja'
        return answer,lang
    if text=='dịch tiếng trung quốc':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng trung'
        lang='zh-CN'
        return answer,lang
    if text=='dịch tiếng nga':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng nga'
        lang='ru'
        return answer,lang
    if text=='dịch tiếng bồ đào nha':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng bồ đào nha'
        lang='pt'
        return answer,lang
    if text=='dịch tiếng pháp':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng pháp'
        lang='fr'
        return answer,lang
    if text=='dịch tiếng ấn độ':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng ấn độ'
        lang='hi'
        return answer,lang
    if text=='dịch tiếng thái lan':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng thái lan'
        lang='th'
        return answer,lang
    if text=='dịch tiếng hàn quốc':
        answer='oke tôi sẽ dịch câu tiếp theo của bạn sang tiếng hàn'
        lang='ko'
        return answer,lang
def command_type3(text):
    answer=calculate(text)
    return answer
def command_type4(answer):
    with open('current_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    l=len(data)
    if l>0:
        if answer=='câu hỏi trước của bạn là':
            return data[l-1]['question']
        if answer=='câu trả lời trước là':
            return data[l-1]['answer']