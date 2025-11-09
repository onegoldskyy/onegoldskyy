/**
 * 생선 바르기 게임 - 메인 게임 로직
 */
class FishFilletingGame {
    constructor() {
        // 캔버스 설정
        this.canvas = document.getElementById('game-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.setupCanvas();

        // UI 컨트롤러
        this.ui = new UIController();

        // 게임 상태
        this.gameState = 'idle'; // idle, playing, finished
        this.fish = null;
        this.difficulty = 'normal';

        // 점수 관련
        this.score = 0;
        this.totalCuts = 0;
        this.accurateCuts = 0;

        // 타이머 관련
        this.timeLimit = null;
        this.remainingTime = null;
        this.timerInterval = null;
        this.startTime = null;

        // 드래그 관련
        this.isDragging = false;
        this.dragPath = [];
        this.lastCheckPoint = null;

        // 이벤트 리스너 설정
        this.setupEventListeners();

        // 초기 화면
        this.ui.showScreen('main');
    }

    /**
     * 캔버스 크기 설정
     */
    setupCanvas() {
        const container = this.canvas.parentElement;
        const maxWidth = 600;
        const maxHeight = 400;

        this.canvas.width = Math.min(maxWidth, container.clientWidth - 40);
        this.canvas.height = maxHeight;
    }

    /**
     * 이벤트 리스너 설정
     */
    setupEventListeners() {
        // 메인 화면 버튼들
        document.getElementById('start-btn').addEventListener('click', () => {
            this.startGame();
        });

        document.getElementById('records-btn').addEventListener('click', () => {
            this.ui.showRecords();
        });

        // 결과 화면 버튼들
        document.getElementById('retry-btn').addEventListener('click', () => {
            this.startGame();
        });

        document.getElementById('home-btn').addEventListener('click', () => {
            this.ui.showScreen('main');
        });

        // 최고 기록 화면 버튼
        document.getElementById('records-back-btn').addEventListener('click', () => {
            this.ui.showScreen('main');
        });

        // 캔버스 마우스 이벤트 (데스크톱)
        this.canvas.addEventListener('mousedown', (e) => this.handleDragStart(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleDragMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleDragEnd(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleDragEnd(e));

        // 캔버스 터치 이벤트 (모바일)
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.handleDragStart(e.touches[0]);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.handleDragMove(e.touches[0]);
        });
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.handleDragEnd(e);
        });

        // 윈도우 리사이즈
        window.addEventListener('resize', () => {
            if (this.gameState === 'playing') {
                this.setupCanvas();
                this.render();
            }
        });
    }

    /**
     * 게임 시작
     */
    startGame() {
        // 난이도 가져오기
        this.difficulty = this.ui.getSelectedDifficulty();

        // 타이머 설정
        switch(this.difficulty) {
            case 'easy':
                this.timeLimit = null; // 무제한
                break;
            case 'normal':
                this.timeLimit = 60;
                break;
            case 'hard':
                this.timeLimit = 45;
                break;
        }

        // 게임 상태 초기화
        this.gameState = 'playing';
        this.score = 0;
        this.totalCuts = 0;
        this.accurateCuts = 0;
        this.remainingTime = this.timeLimit;
        this.startTime = Date.now();

        // 생선 생성
        this.fish = new Fish(this.canvas.width, this.canvas.height, this.difficulty);

        // UI 업데이트
        this.ui.showScreen('game');
        this.ui.updateScore(this.score);
        this.ui.updateTimer(this.remainingTime);
        this.ui.updateProgress(0, this.fish.getTotalCount());

        // 타이머 시작
        if (this.timeLimit !== null) {
            this.startTimer();
        }

        // 렌더링 시작
        this.render();
    }

    /**
     * 타이머 시작
     */
    startTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
        }

        this.timerInterval = setInterval(() => {
            this.remainingTime--;
            this.ui.updateTimer(this.remainingTime);

            if (this.remainingTime <= 0) {
                this.endGame();
            }
        }, 1000);
    }

    /**
     * 드래그 시작
     */
    handleDragStart(e) {
        if (this.gameState !== 'playing') return;

        this.isDragging = true;
        this.dragPath = [];

        const point = this.getCanvasPoint(e);
        this.dragPath.push(point);
        this.lastCheckPoint = point;
    }

    /**
     * 드래그 중
     */
    handleDragMove(e) {
        if (!this.isDragging || this.gameState !== 'playing') return;

        const point = this.getCanvasPoint(e);
        this.dragPath.push(point);

        // 드래그 경로를 따라 가시 제거 체크
        this.checkBoneRemovalAlongPath(point);

        this.render();
    }

    /**
     * 드래그 종료
     */
    handleDragEnd(e) {
        if (this.gameState !== 'playing') return;

        this.isDragging = false;
        this.dragPath = [];
        this.lastCheckPoint = null;
        this.totalCuts++;

        this.render();
    }

    /**
     * 캔버스 좌표 계산
     */
    getCanvasPoint(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
            y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
        };
    }

    /**
     * 드래그 경로를 따라 가시 제거 체크
     */
    checkBoneRemovalAlongPath(currentPoint) {
        if (!this.lastCheckPoint) {
            this.lastCheckPoint = currentPoint;
            return;
        }

        // 마지막 체크 포인트부터 현재 포인트까지의 경로를 따라 체크
        const dx = currentPoint.x - this.lastCheckPoint.x;
        const dy = currentPoint.y - this.lastCheckPoint.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const steps = Math.ceil(distance / 2); // 2픽셀마다 체크

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const x = this.lastCheckPoint.x + dx * t;
            const y = this.lastCheckPoint.y + dy * t;

            if (this.fish.checkBoneRemoval(x, y)) {
                this.accurateCuts++;
                this.updateScore();
                this.updateProgress();
                this.checkGameCompletion();
            }
        }

        this.lastCheckPoint = currentPoint;
    }

    /**
     * 점수 업데이트
     */
    updateScore() {
        const removedCount = this.fish.getRemovedCount();
        this.score = removedCount * 100;
        this.ui.updateScore(this.score);
    }

    /**
     * 진행도 업데이트
     */
    updateProgress() {
        const removedCount = this.fish.getRemovedCount();
        const totalCount = this.fish.getTotalCount();
        this.ui.updateProgress(removedCount, totalCount);
    }

    /**
     * 게임 완료 체크
     */
    checkGameCompletion() {
        if (this.fish.allBonesRemoved()) {
            setTimeout(() => {
                this.endGame();
            }, 500); // 약간의 딜레이 후 종료
        }
    }

    /**
     * 게임 종료
     */
    endGame() {
        if (this.gameState !== 'playing') return;

        this.gameState = 'finished';

        // 타이머 정지
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }

        // 최종 점수 계산
        const removedCount = this.fish.getRemovedCount();
        const totalCount = this.fish.getTotalCount();

        // 기본 점수
        let finalScore = removedCount * 100;

        // 정확도 보너스
        const accuracy = this.totalCuts > 0 ? (this.accurateCuts / this.totalCuts) * 100 : 0;
        const accuracyBonus = (accuracy / 100) * 500;
        finalScore += accuracyBonus;

        // 시간 보너스 (시간 제한이 있는 경우)
        if (this.timeLimit !== null && this.remainingTime > 0) {
            const timeBonus = this.remainingTime * 10;
            finalScore += timeBonus;
        }

        // 소요 시간 계산
        const timeTaken = this.timeLimit !== null
            ? this.timeLimit - this.remainingTime
            : Math.floor((Date.now() - this.startTime) / 1000);

        // 결과 저장
        this.ui.saveRecord(this.difficulty, finalScore, accuracy, timeTaken);

        // 결과 화면 표시
        this.ui.showResult({
            score: finalScore,
            accuracy: accuracy,
            timeTaken: timeTaken
        });
    }

    /**
     * 렌더링
     */
    render() {
        // 캔버스 클리어
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (this.gameState !== 'playing') return;

        // 생선 그리기
        if (this.fish) {
            this.fish.draw(this.ctx);
            this.fish.drawBones(this.ctx);
        }

        // 드래그 경로 그리기
        if (this.isDragging && this.dragPath.length > 1) {
            this.ctx.strokeStyle = '#667eea';
            this.ctx.lineWidth = 3;
            this.ctx.lineCap = 'round';
            this.ctx.lineJoin = 'round';
            this.ctx.globalAlpha = 0.5;

            this.ctx.beginPath();
            this.ctx.moveTo(this.dragPath[0].x, this.dragPath[0].y);

            for (let i = 1; i < this.dragPath.length; i++) {
                this.ctx.lineTo(this.dragPath[i].x, this.dragPath[i].y);
            }

            this.ctx.stroke();
            this.ctx.globalAlpha = 1.0;
        }

        // 게임이 진행 중이면 계속 렌더링
        if (this.gameState === 'playing') {
            requestAnimationFrame(() => this.render());
        }
    }
}

// 게임 시작
let game;
window.addEventListener('DOMContentLoaded', () => {
    game = new FishFilletingGame();
});
