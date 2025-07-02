from flask import Blueprint, render_template, redirect, url_for, request, flash, send_from_directory
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from models import db, User, FileHistory
from detect import detect_image, detect_video
import os
import uuid
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import json
from threading import Thread

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'results'

# Учетные данные GigaChat
GIGACHAT_CREDENTIALS = "MTZkMzYwMjEtMjM2Yy00NzQ2LWFiMjEtNjIwZTkwYWNmYzAyOjc3YjE1NmQ1LTllNWQtNDI3YS05M2EyLTQ4YTc0MWUwODA5Zg=="

def generate_llm_response(prompt, detection_result):
    try:
        try:
            giga = GigaChat(
                credentials=GIGACHAT_CREDENTIALS,
                verify_ssl_certs=False
            )
        except Exception as auth_error:
            print(f"Ошибка аутентификации GigaChat: {str(auth_error)}")
            return "Ошибка аутентификации GigaChat. Проверьте правильность учетных данных"
        
        system_prompt = """
        Ты — эксперт по оценке состояния дорожного покрытия. На основе результата детекции классифицируй состояние дороги по следующим классам:
        0: D00 — Продольные трещины
        1: D01 — Подтип продольных трещин
        2: D10 — Поперечные трещины
        3: D11 — Подтип поперечных трещин
        4: D20 — Сетчатые трещины
        5: D40 — Выбоины
        6: D43 — Большие выбоины
        7: D44 — Глубокие выбоины
        8: D50 — Колеи/деформации/впадины
        9: Repair area — Зона ремонта
        
        Обрати внимание на количество каждого типа дефекта. Чем больше дефектов одного типа, тем серьезнее проблема.
        Учитывай следующее:
        - Если обнаружено более 5 дефектов одного типа - это критическая ситуация
        - Если обнаружено 3-5 дефектов одного типа - это серьезная проблема
        - Если обнаружено 1-2 дефекта одного типа - это требует внимания
        
        Предложи конкретные рекомендации по ремонту, учитывая количество и типы найденных дефектов, только в ответе не говори сколько было обнаружено, просто говори какого типа проблема - критическая, серьёзная или требует внимания или если нет дефектов, то ничего делать не нужно!
        Не говори только про количество, остальное всё можешь говорить!
        Если дефектов не обнаружено - укажи, что дорожное покрытие в хорошем состоянии.
        
        Формат ответа:
        1. Общая оценка состояния дороги
        2. Детальный анализ найденных дефектов
        3. Рекомендации по ремонту
        """
        
        chat = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system_prompt),
                Messages(role=MessagesRole.USER, content=f"Результат детекции: {detection_result}")
            ]
        )
        
        try:
            response = giga.chat(chat)
            return response.choices[0].message.content
        except Exception as chat_error:
            print(f"Ошибка при обращении к GigaChat API: {str(chat_error)}")
            return "Ошибка при обращении к GigaChat API. Проверьте подключение к интернету и доступность сервиса."
        
    except Exception as e:
        print(f"Неожиданная ошибка: {str(e)}")
        return f"Временный ответ: Анализ дефектов дорожного покрытия: {detection_result}. Рекомендации: Провести ремонт выбоин и трещин."

@main.route('/')
def index():
    return redirect(url_for('main.login'))

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Неверные имя пользователя или пароль.')
    return render_template('login.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        hashed_pw = generate_password_hash(request.form['password'])
        new_user = User(username=request.form['username'], password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('main.login'))
    return render_template('register.html')

@main.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    result_text = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            ext = file.filename.rsplit('.', 1)[-1].lower()
            filename = f"{uuid.uuid4().hex}.{ext}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            if ext in ['jpg', 'jpeg', 'png']:
                result_text, processed_path = detect_image(upload_path, RESULT_FOLDER)
            elif ext in ['mp4', 'avi', 'mov']:
                result_text, processed_path = detect_video(upload_path, RESULT_FOLDER)
            else:
                flash("Неподдерживаемый формат файла.")
                return redirect(url_for('main.dashboard'))

            processed_filename = os.path.basename(processed_path)

            history = FileHistory(
                user_id=current_user.id,
                filename=filename,
                result=result_text,
                processed_path=processed_filename
            )
            db.session.add(history)
            db.session.commit()

    files = FileHistory.query.filter_by(user_id=current_user.id).order_by(FileHistory.uploaded_at.desc()).all()
    return render_template('dashboard.html', user=current_user, files=files, result=result_text)


@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))

@main.route('/file/<int:file_id>', methods=['GET', 'POST'])
@login_required
def file_details(file_id):
    file = FileHistory.query.get_or_404(file_id)
    if file.user_id != current_user.id:
        flash("Нет доступа к этому файлу.")
        return redirect(url_for('main.dashboard'))

    llm_response = generate_llm_response("", file.result)
    return render_template('file_details.html', file_history=file, llm_response=llm_response)

@main.route('/download/<path:filename>')
@login_required
def download_file(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(RESULT_FOLDER, filename)

    if os.path.exists(file_path):
        return send_from_directory(RESULT_FOLDER, filename, as_attachment=False)
    else:
        flash("Файл не найден.")
        return redirect(url_for('main.dashboard'))

@main.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('Нет выбранного файла', 'error')
        return redirect(url_for('main.index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Нет выбранного файла', 'error')
        return redirect(url_for('main.index'))
    
    if file:
        # Сохраняем оригинальный файл
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Генерируем уникальное имя для обработанного файла
        processed_filename = f"processed_{uuid.uuid4().hex}.jpg"
        processed_path = os.path.join(RESULT_FOLDER, processed_filename)
        
        # Создаем запись в базе данных
        file_history = FileHistory(
            filename=filename,  # Сохраняем оригинальное имя файла
            processed_path=processed_filename,  # Сохраняем имя обработанного файла
            result="Результат детекции будет добавлен позже",
            user_id=current_user.id
        )
        db.session.add(file_history)
        db.session.commit()
        
        # Запускаем обработку в отдельном потоке
        thread = Thread(target=process_file, args=(file_path, processed_path, file_history.id))
        thread.start()
        
        flash('Файл успешно загружен и отправлен на обработку', 'success')
        return redirect(url_for('main.index'))
    
    flash('Ошибка при загрузке файла', 'error')
    return redirect(url_for('main.index'))
