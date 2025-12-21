/**
 * 2026 Red Horse Time Capsule - Main JavaScript
 * Mobile-First Interactive Features
 */

// ============================================
// Application State
// ============================================
const AppState = {
    resolutions: [],
    STORAGE_KEY: 'redHorse2026_resolutions'
};

// ============================================
// Sample Resolutions (Initial Data)
// ============================================
const SAMPLE_RESOLUTIONS = [
    {
        id: Date.now() + 1,
        name: '희망이',
        resolution: '2026년에는 건강한 몸과 마음을 위해 매일 아침 30분씩 운동하고, 긍정적인 마음가짐으로 하루를 시작하겠습니다!',
        email: '',
        date: new Date().toISOString()
    },
    {
        id: Date.now() + 2,
        name: '도전자',
        resolution: '새로운 기술을 배우고 성장하여, 1년 안에 웹 개발자로 취업하는 것이 저의 목표입니다. 매일 꾸준히 공부하겠습니다.',
        email: '',
        date: new Date().toISOString()
    },
    {
        id: Date.now() + 3,
        name: '익명',
        resolution: '가족과 더 많은 시간을 보내고, 소중한 사람들에게 사랑한다는 말을 자주 하겠습니다. 💕',
        email: '',
        date: new Date().toISOString()
    },
    {
        id: Date.now() + 4,
        name: '꿈나무',
        resolution: '책을 월 2권 이상 읽고, 독서 노트를 작성하여 지식과 통찰력을 쌓아가겠습니다.',
        email: '',
        date: new Date().toISOString()
    },
    {
        id: Date.now() + 5,
        name: '절약왕',
        resolution: '매달 수입의 20%를 저축하여 1년 뒤에는 여행 자금을 모으고, 계획했던 유럽 여행을 떠나겠습니다!',
        email: '',
        date: new Date().toISOString()
    }
];

// ============================================
// DOM Elements
// ============================================
const DOM = {
    form: null,
    nameInput: null,
    resolutionInput: null,
    emailInput: null,
    charCount: null,
    giftCount: null,
    galleryGrid: null,
    modal: null,
    modalTitle: null,
    modalResolution: null,
    modalDate: null,
    modalClose: null,
    modalBackdrop: null,
    successMessage: null
};

// ============================================
// Initialize DOM References
// ============================================
function initDOM() {
    DOM.form = document.getElementById('resolutionForm');
    DOM.nameInput = document.getElementById('userName');
    DOM.resolutionInput = document.getElementById('resolution');
    DOM.emailInput = document.getElementById('userEmail');
    DOM.charCount = document.getElementById('charCount');
    DOM.giftCount = document.getElementById('giftCount').querySelector('.count');
    DOM.galleryGrid = document.getElementById('galleryGrid');
    DOM.modal = document.getElementById('resolutionModal');
    DOM.modalTitle = document.getElementById('modalTitle');
    DOM.modalResolution = document.getElementById('modalResolution');
    DOM.modalDate = document.getElementById('modalDate');
    DOM.modalClose = DOM.modal.querySelector('.modal-close');
    DOM.modalBackdrop = DOM.modal.querySelector('.modal-backdrop');
    DOM.successMessage = document.getElementById('successMessage');
}

// ============================================
// Local Storage Management
// ============================================
function loadResolutions() {
    try {
        const stored = localStorage.getItem(AppState.STORAGE_KEY);
        if (stored) {
            AppState.resolutions = JSON.parse(stored);
        } else {
            // Initialize with sample data
            AppState.resolutions = SAMPLE_RESOLUTIONS;
            saveResolutions();
        }
    } catch (error) {
        console.error('Error loading resolutions:', error);
        AppState.resolutions = SAMPLE_RESOLUTIONS;
    }
}

function saveResolutions() {
    try {
        localStorage.setItem(AppState.STORAGE_KEY, JSON.stringify(AppState.resolutions));
    } catch (error) {
        console.error('Error saving resolutions:', error);
    }
}

// ============================================
// Resolution Management
// ============================================
function addResolution(name, resolution, email) {
    const newResolution = {
        id: Date.now(),
        name: name.trim() || '익명',
        resolution: resolution.trim(),
        email: email.trim(),
        date: new Date().toISOString()
    };

    AppState.resolutions.unshift(newResolution);
    saveResolutions();
    return newResolution;
}

// ============================================
// Date Formatting
// ============================================
function formatDate(isoString) {
    const date = new Date(isoString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');

    return `${year}.${month}.${day} ${hours}:${minutes}`;
}

// ============================================
// Gallery Rendering
// ============================================
function renderGallery() {
    if (!DOM.galleryGrid) return;

    DOM.galleryGrid.innerHTML = '';

    if (AppState.resolutions.length === 0) {
        DOM.galleryGrid.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: white; opacity: 0.7;">
                <p style="font-size: 1.2rem;">아직 다짐이 없습니다</p>
                <p style="font-size: 0.9rem; margin-top: 8px;">첫 번째 다짐을 작성해보세요!</p>
            </div>
        `;
        return;
    }

    AppState.resolutions.forEach(res => {
        const card = createResolutionCard(res);
        DOM.galleryGrid.appendChild(card);
    });

    updateGiftCount();
}

function createResolutionCard(resolution) {
    const card = document.createElement('div');
    card.className = 'resolution-card';
    card.setAttribute('data-id', resolution.id);
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');

    const truncatedResolution = resolution.resolution.length > 100
        ? resolution.resolution.substring(0, 100) + '...'
        : resolution.resolution;

    card.innerHTML = `
        <div class="card-header">
            <div class="card-author">${escapeHTML(resolution.name)}</div>
            <div class="card-icon">🎁</div>
        </div>
        <div class="card-resolution">
            ${escapeHTML(truncatedResolution)}
        </div>
        <div class="card-footer">
            <div class="card-date">${formatDate(resolution.date)}</div>
            <div class="card-read-more">자세히 보기 →</div>
        </div>
    `;

    // Click and keyboard event
    card.addEventListener('click', () => openModal(resolution));
    card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            openModal(resolution);
        }
    });

    return card;
}

// ============================================
// Modal Management
// ============================================
function openModal(resolution) {
    if (!DOM.modal) return;

    DOM.modalTitle.textContent = `${resolution.name}님의 다짐`;
    DOM.modalResolution.textContent = resolution.resolution;
    DOM.modalDate.textContent = `작성일: ${formatDate(resolution.date)}`;

    DOM.modal.classList.add('active');
    document.body.style.overflow = 'hidden';

    // Focus management
    DOM.modalClose.focus();
}

function closeModal() {
    if (!DOM.modal) return;

    DOM.modal.classList.remove('active');
    document.body.style.overflow = '';
}

// ============================================
// Form Handling
// ============================================
function handleFormSubmit(e) {
    e.preventDefault();

    const name = DOM.nameInput.value;
    const resolution = DOM.resolutionInput.value;
    const email = DOM.emailInput.value;

    // Validation
    if (!resolution.trim()) {
        alert('다짐을 작성해주세요!');
        DOM.resolutionInput.focus();
        return;
    }

    if (resolution.trim().length < 10) {
        alert('다짐은 최소 10자 이상 작성해주세요!');
        DOM.resolutionInput.focus();
        return;
    }

    // Email validation (if provided)
    if (email && !isValidEmail(email)) {
        alert('올바른 이메일 주소를 입력해주세요!');
        DOM.emailInput.focus();
        return;
    }

    // Add resolution
    addResolution(name, resolution, email);

    // Show success message
    showSuccessMessage();

    // Re-render gallery
    renderGallery();

    // Reset form
    DOM.form.reset();
    updateCharCount();

    // Scroll to gallery
    setTimeout(() => {
        DOM.galleryGrid.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 1000);
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// ============================================
// Character Counter
// ============================================
function updateCharCount() {
    if (!DOM.resolutionInput || !DOM.charCount) return;

    const count = DOM.resolutionInput.value.length;
    DOM.charCount.textContent = count;

    // Color feedback
    if (count >= 190) {
        DOM.charCount.style.color = '#DC143C';
        DOM.charCount.style.fontWeight = '700';
    } else {
        DOM.charCount.style.color = '';
        DOM.charCount.style.fontWeight = '';
    }
}

// ============================================
// Gift Count Update
// ============================================
function updateGiftCount() {
    if (!DOM.giftCount) return;
    DOM.giftCount.textContent = AppState.resolutions.length;
}

// ============================================
// Success Message
// ============================================
function showSuccessMessage() {
    if (!DOM.successMessage) return;

    DOM.successMessage.classList.add('show');

    setTimeout(() => {
        DOM.successMessage.classList.remove('show');
    }, 3000);
}

// ============================================
// Utility Functions
// ============================================
function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Event Listeners
// ============================================
function attachEventListeners() {
    // Form submission
    if (DOM.form) {
        DOM.form.addEventListener('submit', handleFormSubmit);
    }

    // Character counter
    if (DOM.resolutionInput) {
        DOM.resolutionInput.addEventListener('input', updateCharCount);
    }

    // Modal close
    if (DOM.modalClose) {
        DOM.modalClose.addEventListener('click', closeModal);
    }

    if (DOM.modalBackdrop) {
        DOM.modalBackdrop.addEventListener('click', closeModal);
    }

    // Modal keyboard close
    if (DOM.modal) {
        DOM.modal.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    }

    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Escape key to close modal
        if (e.key === 'Escape' && DOM.modal.classList.contains('active')) {
            closeModal();
        }
    });
}

// ============================================
// Touch Event Optimizations
// ============================================
function initTouchOptimizations() {
    // Prevent double-tap zoom on buttons
    const buttons = document.querySelectorAll('button, .resolution-card');
    buttons.forEach(button => {
        button.addEventListener('touchend', (e) => {
            // Prevent default only for specific elements
            if (button.tagName === 'BUTTON') {
                e.preventDefault();
            }
        }, { passive: false });
    });
}

// ============================================
// Initialize Application
// ============================================
function init() {
    // Initialize DOM references
    initDOM();

    // Load resolutions from storage
    loadResolutions();

    // Render initial gallery
    renderGallery();

    // Attach event listeners
    attachEventListeners();

    // Touch optimizations
    initTouchOptimizations();

    // Initialize character counter
    updateCharCount();

    console.log('🐴 2026 Red Horse Time Capsule initialized!');
    console.log(`📦 ${AppState.resolutions.length} resolutions loaded`);
}

// ============================================
// Start Application
// ============================================
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// ============================================
// Export for debugging (optional)
// ============================================
window.RedHorseApp = {
    state: AppState,
    renderGallery,
    loadResolutions,
    saveResolutions
};
