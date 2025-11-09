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
        ctx.save();
        ctx.translate(this.x, this.y);

        // 그림자 효과
        ctx.shadowColor = 'rgba(0, 0, 0, 0.15)';
        ctx.shadowBlur = 15;
        ctx.shadowOffsetX = 4;
        ctx.shadowOffsetY = 4;

        // 꼬리 지느러미 (먼저 그려서 몸통 뒤에 위치)
        ctx.fillStyle = '#FF9AA2';
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;

        ctx.beginPath();
        ctx.moveTo(-this.width/2, -15);
        ctx.lineTo(-this.width/2 - 40, -35);
        ctx.lineTo(-this.width/2 - 35, 0);
        ctx.lineTo(-this.width/2 - 40, 35);
        ctx.lineTo(-this.width/2, 15);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // 꼬리 지느러미 디테일 선
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(-this.width/2 - 10, -15);
        ctx.lineTo(-this.width/2 - 35, 0);
        ctx.moveTo(-this.width/2 - 10, 15);
        ctx.lineTo(-this.width/2 - 35, 0);
        ctx.stroke();

        // 생선 몸통 (타원형)
        const gradient = ctx.createLinearGradient(-this.width/2, -this.height/2, this.width/2, this.height/2);
        gradient.addColorStop(0, '#FFB3BA');
        gradient.addColorStop(0.5, '#FFCCD5');
        gradient.addColorStop(1, '#FFB3BA');

        ctx.fillStyle = gradient;
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 4;

        ctx.beginPath();
        ctx.ellipse(0, 0, this.width/2, this.height/2, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // 배 부분 (더 밝은 색)
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.beginPath();
        ctx.ellipse(0, this.height/6, this.width/2.5, this.height/4, 0, 0, Math.PI * 2);
        ctx.fill();

        // 비늘 패턴
        ctx.strokeStyle = 'rgba(255, 150, 160, 0.3)';
        ctx.lineWidth = 1.5;
        for (let i = -this.width/3; i < this.width/3; i += 20) {
            for (let j = -this.height/4; j < this.height/4; j += 20) {
                ctx.beginPath();
                ctx.arc(i, j - 10, 8, 0, Math.PI * 2);
                ctx.stroke();
            }
        }

        ctx.shadowColor = 'transparent';

        // 등 지느러미
        ctx.fillStyle = '#FF9AA2';
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 3;

        ctx.beginPath();
        ctx.moveTo(-30, -this.height/2);
        ctx.lineTo(-20, -this.height/2 - 25);
        ctx.lineTo(0, -this.height/2 - 30);
        ctx.lineTo(20, -this.height/2 - 25);
        ctx.lineTo(30, -this.height/2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // 배 지느러미
        ctx.beginPath();
        ctx.moveTo(10, this.height/2);
        ctx.lineTo(5, this.height/2 + 15);
        ctx.lineTo(20, this.height/2 + 10);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // 생선 눈
        const eyeX = this.width/3;
        const eyeY = -this.height/5;

        // 눈 흰자
        ctx.fillStyle = 'white';
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, 14, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // 눈동자
        ctx.fillStyle = '#2C3E50';
        ctx.beginPath();
        ctx.arc(eyeX + 2, eyeY, 7, 0, Math.PI * 2);
        ctx.fill();

        // 눈 하이라이트
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.arc(eyeX + 4, eyeY - 2, 3, 0, Math.PI * 2);
        ctx.fill();

        // 생선 입 (미소)
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.arc(this.width/2 - 15, 5, 10, 0.1, Math.PI * 0.8);
        ctx.stroke();

        // 아가미
        ctx.strokeStyle = '#FF6B6B';
        ctx.lineWidth = 2;
        for (let i = 0; i < 3; i++) {
            ctx.beginPath();
            ctx.arc(this.width/6 - i * 8, 0, 25, Math.PI * 0.6, Math.PI * 1.4);
            ctx.stroke();
        }

        ctx.restore();
    }

    /**
     * 가시(뼈) 그리기
     */
    drawBones(ctx) {
        this.bones.forEach(bone => {
            if (bone.removed) {
                // 제거된 가시는 큰 체크 마크로 표시
                if (bone.marked) {
                    ctx.save();

                    // 배경 원
                    ctx.fillStyle = 'rgba(76, 175, 80, 0.2)';
                    ctx.beginPath();
                    ctx.arc(bone.x, bone.y, 15, 0, Math.PI * 2);
                    ctx.fill();

                    // 체크 마크
                    ctx.strokeStyle = '#4CAF50';
                    ctx.lineWidth = 4;
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';
                    ctx.beginPath();
                    ctx.moveTo(bone.x - 8, bone.y);
                    ctx.lineTo(bone.x - 2, bone.y + 8);
                    ctx.lineTo(bone.x + 10, bone.y - 8);
                    ctx.stroke();

                    ctx.restore();
                }
            } else {
                // 가시 그리기 (더 눈에 잘 띄게)
                ctx.save();

                // 가시 배경 (강조용)
                ctx.fillStyle = 'rgba(255, 107, 107, 0.15)';
                ctx.beginPath();
                ctx.arc(bone.x, bone.y, bone.radius + 8, 0, Math.PI * 2);
                ctx.fill();

                // 가시 몸통 (흰색 뼈 모양)
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 5;
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(bone.x, bone.y - 15);
                ctx.lineTo(bone.x, bone.y + 15);
                ctx.stroke();

                // 가시 옆가지 (더 길고 명확하게)
                ctx.lineWidth = 4;
                ctx.beginPath();
                ctx.moveTo(bone.x - 8, bone.y - 8);
                ctx.lineTo(bone.x + 8, bone.y - 8);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(bone.x - 8, bone.y);
                ctx.lineTo(bone.x + 8, bone.y);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(bone.x - 8, bone.y + 8);
                ctx.lineTo(bone.x + 8, bone.y + 8);
                ctx.stroke();

                // 가시 위치 표시 (빨간 원 - 더 크게)
                ctx.shadowColor = 'rgba(255, 71, 87, 0.4)';
                ctx.shadowBlur = 8;

                ctx.fillStyle = '#FF4757';
                ctx.strokeStyle = '#DC143C';
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.arc(bone.x, bone.y, bone.radius + 2, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.shadowColor = 'transparent';

                // 내부 하이라이트
                ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                ctx.beginPath();
                ctx.arc(bone.x - 3, bone.y - 3, 4, 0, Math.PI * 2);
                ctx.fill();

                // 경고 텍스트 (선택적)
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 12px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('!', bone.x, bone.y);

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
