/**
 * 생선 클래스
 * 생선의 모양, 가시(뼈) 위치 등을 관리
 */
class Fish {
    constructor(canvasWidth, canvasHeight, difficulty) {
        this.canvasWidth = canvasWidth;
        this.canvasHeight = canvasHeight;
        this.difficulty = difficulty;

        // 생선 크기 및 위치
        this.x = canvasWidth / 2;
        this.y = canvasHeight / 2;
        this.width = 300;
        this.height = 120;

        // 난이도별 가시 개수
        this.boneCount = this.getBoneCount();
        this.bones = [];
        this.generateBones();
    }

    /**
     * 난이도별 가시 개수 반환
     */
    getBoneCount() {
        switch(this.difficulty) {
            case 'easy': return 5;
            case 'normal': return 7;
            case 'hard': return 10;
            default: return 7;
        }
    }

    /**
     * 가시(뼈) 생성
     * 생선 몸통 내부에 랜덤하게 배치
     */
    generateBones() {
        this.bones = [];
        const spacing = this.width / (this.boneCount + 1);

        for (let i = 0; i < this.boneCount; i++) {
            // 생선 중앙선을 따라 배치하되, 약간의 랜덤성 추가
            const x = this.x - this.width/2 + spacing * (i + 1);
            const y = this.y + (Math.random() - 0.5) * (this.height * 0.4);

            this.bones.push({
                x: x,
                y: y,
                radius: 8,
                removed: false,
                marked: false // 체크 마크 표시용
            });
        }
    }

    /**
     * 생선 그리기
     */
    draw(ctx) {
        // 생선 몸통 (타원형)
        ctx.save();
        ctx.translate(this.x, this.y);

        // 그림자 효과
        ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
        ctx.shadowBlur = 10;
        ctx.shadowOffsetX = 3;
        ctx.shadowOffsetY = 3;

        // 생선 몸통
        ctx.fillStyle = '#87CEEB';
        ctx.strokeStyle = '#4682B4';
        ctx.lineWidth = 3;

        ctx.beginPath();
        ctx.ellipse(0, 0, this.width/2, this.height/2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.shadowColor = 'transparent';

        // 생선 눈
        const eyeX = this.width/3;
        const eyeY = -this.height/6;

        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(eyeX + 2, eyeY, 6, 0, Math.PI * 2);
        ctx.fill();

        // 생선 입
        ctx.strokeStyle = '#4682B4';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(this.width/2 - 10, 0, 8, 0.2, Math.PI - 0.2);
        ctx.stroke();

        // 지느러미 (꼬리)
        ctx.fillStyle = '#5F9EA0';
        ctx.strokeStyle = '#4682B4';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(-this.width/2, -10);
        ctx.lineTo(-this.width/2 - 30, -25);
        ctx.lineTo(-this.width/2 - 30, 25);
        ctx.lineTo(-this.width/2, 10);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // 등 지느러미
        ctx.beginPath();
        ctx.moveTo(-20, -this.height/2);
        ctx.lineTo(-10, -this.height/2 - 20);
        ctx.lineTo(20, -this.height/2 - 20);
        ctx.lineTo(30, -this.height/2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        ctx.restore();
    }

    /**
     * 가시(뼈) 그리기
     */
    drawBones(ctx) {
        this.bones.forEach(bone => {
            if (bone.removed) {
                // 제거된 가시는 체크 마크로 표시
                if (bone.marked) {
                    ctx.strokeStyle = '#4CAF50';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(bone.x - 8, bone.y);
                    ctx.lineTo(bone.x - 2, bone.y + 6);
                    ctx.lineTo(bone.x + 8, bone.y - 8);
                    ctx.stroke();
                }
            } else {
                // 가시 그리기 (빨간 점과 선)
                ctx.save();

                // 가시 몸통 (흰색 뼈 모양)
                ctx.strokeStyle = '#FFE0E0';
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.moveTo(bone.x, bone.y - 12);
                ctx.lineTo(bone.x, bone.y + 12);
                ctx.stroke();

                // 가시 옆가지
                ctx.beginPath();
                ctx.moveTo(bone.x - 6, bone.y - 6);
                ctx.lineTo(bone.x + 6, bone.y - 6);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(bone.x - 6, bone.y + 6);
                ctx.lineTo(bone.x + 6, bone.y + 6);
                ctx.stroke();

                // 가시 위치 표시 (빨간 원)
                ctx.fillStyle = '#FF6B6B';
                ctx.strokeStyle = '#FF4757';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(bone.x, bone.y, bone.radius, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                // 내부 하이라이트
                ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.beginPath();
                ctx.arc(bone.x - 2, bone.y - 2, 3, 0, Math.PI * 2);
                ctx.fill();

                ctx.restore();
            }
        });
    }

    /**
     * 특정 위치에 가시가 있는지 확인하고 제거
     * @param {number} x - x 좌표
     * @param {number} y - y 좌표
     * @returns {boolean} - 가시를 제거했는지 여부
     */
    checkBoneRemoval(x, y) {
        for (let bone of this.bones) {
            if (!bone.removed) {
                const distance = Math.sqrt(
                    Math.pow(x - bone.x, 2) + Math.pow(y - bone.y, 2)
                );

                // 가시의 범위 내에 있으면 제거
                if (distance < bone.radius + 5) {
                    bone.removed = true;
                    bone.marked = true;
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * 모든 가시가 제거되었는지 확인
     */
    allBonesRemoved() {
        return this.bones.every(bone => bone.removed);
    }

    /**
     * 제거된 가시 개수 반환
     */
    getRemovedCount() {
        return this.bones.filter(bone => bone.removed).length;
    }

    /**
     * 전체 가시 개수 반환
     */
    getTotalCount() {
        return this.bones.length;
    }
}
