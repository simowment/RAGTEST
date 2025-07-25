class RAGAssistant {
    constructor() {
        this.currentMode = 'vectorbt';
        this.chatHistory = [];
        this.isLoading = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupCodeEditor();
        this.updateUI();
    }

    initializeElements() {
        // Mode buttons
        this.vectorbtModeBtn = document.getElementById('vectorbt-mode');
        this.reviewModeBtn = document.getElementById('review-mode');
        
        // Panels
        this.reviewPanel = document.getElementById('review-panel');
        
        // Code editor
        this.codeEditor = document.getElementById('code-editor');
        this.languageSelect = document.getElementById('language-select');
        this.clearCodeBtn = document.getElementById('clear-code');
        this.pasteCodeBtn = document.getElementById('paste-code');
        
        // Chat elements
        this.chatTitle = document.getElementById('chat-title');
        this.chatMessages = document.getElementById('chat-messages');
        this.questionInput = document.getElementById('question-input');
        this.sendBtn = document.getElementById('send-btn');
        this.clearChatBtn = document.getElementById('clear-chat');
        
        // Loading
        this.loadingEl = document.getElementById('loading');
    }

    setupEventListeners() {
        // Mode switching
        this.vectorbtModeBtn.addEventListener('click', () => this.switchMode('vectorbt'));
        this.reviewModeBtn.addEventListener('click', () => this.switchMode('review'));
        
        // Code actions
        this.clearCodeBtn.addEventListener('click', () => this.clearCode());
        this.pasteCodeBtn.addEventListener('click', () => this.pasteCode());
        this.languageSelect.addEventListener('change', () => this.changeLanguage());
        
        // Chat actions
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        
        // Input handling
        this.questionInput.addEventListener('input', () => this.updateSendButton());
        this.questionInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.questionInput.addEventListener('input', () => this.autoResizeTextarea(this.questionInput));
    }

    setupCodeEditor() {
        // Auto-resize code editor
        this.codeEditor.addEventListener('input', () => {
            this.autoResizeTextarea(this.codeEditor);
            this.updateSendButton();
        });
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update button states
        this.vectorbtModeBtn.classList.toggle('active', mode === 'vectorbt');
        this.reviewModeBtn.classList.toggle('active', mode === 'review');
        
        // Update UI
        this.updateUI();
        this.clearChat();
    }

    updateUI() {
        const isReviewMode = this.currentMode === 'review';
        
        // Show/hide panels
        this.reviewPanel.classList.toggle('hidden', !isReviewMode);
        
        // Update layout
        document.body.classList.toggle('vectorbt-mode', !isReviewMode);
        
        // Update chat title
        this.chatTitle.textContent = isReviewMode ? 
            '🔍 Chat - Review de Code' : 
            '📚 Chat - Documentation VectorBT';
    }

    async clearCode() {
        this.codeEditor.value = '';
        this.updateSendButton();
    }

    async pasteCode() {
        try {
            const text = await navigator.clipboard.readText();
            this.codeEditor.value = text;
            this.autoResizeTextarea(this.codeEditor);
            this.updateSendButton();
        } catch (err) {
            console.error('Erreur lors du collage:', err);
            // Fallback: focus on textarea for manual paste
            this.codeEditor.focus();
        }
    }

    changeLanguage() {
        // This could be extended to change syntax highlighting
        console.log('Language changed to:', this.languageSelect.value);
    }

    updateSendButton() {
        const hasQuestion = this.questionInput.value.trim().length > 0;
        const hasCode = this.currentMode === 'vectorbt' || this.codeEditor.value.trim().length > 0;
        
        this.sendBtn.disabled = !hasQuestion || (this.currentMode === 'review' && !hasCode) || this.isLoading;
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    async sendMessage() {
        if (this.isLoading || this.sendBtn.disabled) return;

        const question = this.questionInput.value.trim();
        if (!question) return;

        // Add user message to chat
        this.addMessage('user', question);
        
        // Clear input
        this.questionInput.value = '';
        this.updateSendButton();
        this.autoResizeTextarea(this.questionInput);

        // Show loading
        this.setLoading(true);

        try {
            let response;
            
            if (this.currentMode === 'vectorbt') {
                response = await this.queryVectorBT(question);
            } else {
                const code = this.codeEditor.value.trim();
                if (!code) {
                    throw new Error('Veuillez fournir du code à analyser');
                }
                response = await this.reviewCode(code, question);
            }

            // Add assistant response
            this.addMessage('assistant', response.response || response.message || 'Réponse reçue');
            
        } catch (error) {
            console.error('Erreur:', error);
            this.addMessage('assistant', `❌ Erreur: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    async queryVectorBT(question) {
        const response = await fetch('/vectorbt/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Erreur lors de la requête');
        }

        return await response.json();
    }

    async reviewCode(code, question) {
        const response = await fetch('/review/code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code, question })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Erreur lors de la review');
        }

        return await response.json();
    }

    addMessage(sender, content) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${sender}`;
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        // Format content with basic markdown-like formatting
        const formattedContent = this.formatMessage(content);
        contentEl.innerHTML = formattedContent;
        
        messageEl.appendChild(contentEl);
        this.chatMessages.appendChild(messageEl);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        // Add to history
        this.chatHistory.push({ sender, content, timestamp: new Date() });
    }

    formatMessage(content) {
        // Basic formatting for code blocks and inline code
        let formatted = content
            .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        return formatted;
    }

    clearChat() {
        this.chatMessages.innerHTML = '';
        this.chatHistory = [];
        
        // Add welcome message
        const welcomeMessage = this.currentMode === 'vectorbt' 
            ? 'Bonjour ! Je peux vous aider avec la documentation VectorBT. Posez-moi vos questions !'
            : 'Bonjour ! Collez votre code et posez vos questions pour une review détaillée.';
            
        this.addMessage('assistant', welcomeMessage);
    }

    setLoading(loading) {
        this.isLoading = loading;
        this.loadingEl.classList.toggle('hidden', !loading);
        this.updateSendButton();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGAssistant();
});