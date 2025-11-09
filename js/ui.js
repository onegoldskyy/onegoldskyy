/**
 * UI 컨트롤러
 * 화면 전환 및 UI 업데이트 담당
 */
class UIController {
    constructor() {
        // 화면 요소들
        this.screens = {
            main: document.getElementById('main-screen'),
            game: document.getElementById('game-screen'),
            result: document.getElementById('result-screen'),
            records: document.getElementById('records-screen')
        };

        // UI 요소들
        this.elements = {
            score: document.getElementById('score'),
            timer: document.getElementById('timer'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            stars: document.getElementById('stars'),
            finalScore: document.getElementById('final-score'),
            accuracy: document.getElementById('accuracy'),
            timeTaken: document.getElementById('time-taken'),
            recordsList: document.getElementById('records-list')
        };

        // 난이도 선택
        this.selectedDifficulty = 'normal';
        this.setupDifficultyButtons();
    }

    /**
     * 난이도 버튼 설정
     */
    setupDifficultyButtons() {
        const difficultyButtons = document.querySelectorAll('.difficulty-btn');
        difficultyButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                difficultyButtons.forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.selectedDifficulty = btn.dataset.difficulty;
            });
        });
    }

    /**
     * 화면 전환
     */
    showScreen(screenName) {
        Object.values(this.screens).forEach(screen => {
            screen.classList.remove('active');
        });
        if (this.screens[screenName]) {
            this.screens[screenName].classList.add('active');
        }
    }

    /**
     * 점수 업데이트
     */
    updateScore(score) {
        this.elements.score.textContent = Math.round(score);
    }

    /**
     * 타이머 업데이트
     */
    updateTimer(seconds) {
        if (seconds === null) {
            this.elements.timer.textContent = '무제한';
        } else {
            this.elements.timer.textContent = `${seconds}초`;
        }
    }

    /**
     * 진행도 바 업데이트
     */
    updateProgress(removed, total) {
        const percentage = (removed / total) * 100;
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = `${removed}/${total}`;
    }

    /**
     * 결과 화면 표시
     */
    showResult(stats) {
        // 별점 계산 (점수 기반)
        let starCount = 1;
        if (stats.score >= 800) starCount = 3;
        else if (stats.score >= 500) starCount = 2;

        // 별 표시
        const starHTML = '⭐'.repeat(starCount) + '☆'.repeat(3 - starCount);
        this.elements.stars.innerHTML = starHTML;

        // 통계 표시
        this.elements.finalScore.textContent = Math.round(stats.score);
        this.elements.accuracy.textContent = `${Math.round(stats.accuracy)}%`;
        this.elements.timeTaken.textContent = `${stats.timeTaken}초`;

        // 결과 화면으로 전환
        this.showScreen('result');
    }

    /**
     * 최고 기록 표시
     */
    showRecords() {
        const records = this.loadRecords();
        const recordsList = this.elements.recordsList;

        if (records.length === 0) {
            recordsList.innerHTML = '<div class="no-records">아직 기록이 없습니다</div>';
        } else {
            recordsList.innerHTML = records.map((record, index) => `
                <div class="record-item">
                    <div>
                        <div class="record-difficulty">${this.getDifficultyText(record.difficulty)}</div>
                        <div style="color: #999; font-size: 0.9em;">${record.date}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.3em; font-weight: bold; color: #667eea;">
                            ${Math.round(record.score)}점
                        </div>
                        <div style="color: #999; font-size: 0.9em;">
                            정확도: ${Math.round(record.accuracy)}%
                        </div>
                    </div>
                </div>
            `).join('');
        }

        this.showScreen('records');
    }

    /**
     * 난이도 텍스트 반환
     */
    getDifficultyText(difficulty) {
        const texts = {
            'easy': '쉬움',
            'normal': '보통',
            'hard': '어려움'
        };
        return texts[difficulty] || difficulty;
    }

    /**
     * 기록 저장
     */
    saveRecord(difficulty, score, accuracy, timeTaken) {
        const records = this.loadRecords();
        const newRecord = {
            difficulty,
            score,
            accuracy,
            timeTaken,
            date: new Date().toLocaleDateString('ko-KR')
        };

        records.push(newRecord);

        // 점수 기준으로 정렬하고 상위 10개만 유지
        records.sort((a, b) => b.score - a.score);
        const topRecords = records.slice(0, 10);

        localStorage.setItem('fishGameRecords', JSON.stringify(topRecords));
    }

    /**
     * 기록 불러오기
     */
    loadRecords() {
        const saved = localStorage.getItem('fishGameRecords');
        return saved ? JSON.parse(saved) : [];
    }

    /**
     * 선택된 난이도 반환
     */
    getSelectedDifficulty() {
        return this.selectedDifficulty;
    }
}
