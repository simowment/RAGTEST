class RAGAssistant {
    constructor() {
        this.currentKnowledgeBase = null;
        this.knowledgeBases = [];
        this.chatHistories = {}; // Separate history for each knowledge base
        this.isLoading = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupCodeEditor();
        this.loadKnowledgeBases();
    }

    initializeElements() {
        // Mode selector
        this.modeSelector = document.getElementById('mode-selector');
        
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
        this.kbStatus = document.getElementById('kb-status');
        
        // Image elements
        this.imageBtn = document.getElementById('image-btn');
        this.imageInput = document.getElementById('image-input');
        this.imagePreviewContainer = document.getElementById('image-preview-container');
        
        // Image storage
        this.selectedImages = [];
    }

    setupEventListeners() {
        // Code actions
        this.clearCodeBtn.addEventListener('click', () => this.clearCode());
        this.pasteCodeBtn.addEventListener('click', () => this.pasteCode());
        this.languageSelect.addEventListener('change', () => this.changeLanguage());
        
        // Chat actions
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        
        // Image actions
        this.imageBtn.addEventListener('click', () => this.imageInput.click());
        this.imageInput.addEventListener('change', (e) => this.handleImageSelect(e));
        
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

    async loadKnowledgeBases() {
        try {
            this.showLoadingOverlay('Chargement des bases de connaissances...');
            
            const response = await fetch('/knowledge-bases');
            const data = await response.json();
            this.knowledgeBases = data.knowledge_bases;
            this.renderKnowledgeBaseSelector();
            
            // Select the first available knowledge base
            const availableKB = this.knowledgeBases.find(kb => kb.available);
            if (availableKB) {
                this.switchKnowledgeBase(availableKB.id);
            }
            
            this.hideLoadingOverlay();
        } catch (error) {
            console.error('Error loading knowledge bases:', error);
            this.addMessage('assistant', '❌ Erreur lors du chargement des bases de connaissances');
            this.hideLoadingOverlay();
        }
    }

    renderKnowledgeBaseSelector() {
        this.modeSelector.innerHTML = this.knowledgeBases.map(kb => {
            const statusIcon = kb.available ? '' : ' ❌';
            const disabled = kb.available ? '' : 'disabled';
            return `
                <button 
                    class="mode-btn ${disabled}" 
                    data-kb-id="${kb.id}"
                    ${disabled ? 'disabled' : ''}
                    title="${kb.description}"
                >
                    ${kb.icon} ${kb.name}${statusIcon}
                </button>
            `;
        }).join('');

        // Add event listeners to knowledge base buttons
        this.modeSelector.querySelectorAll('.mode-btn:not([disabled])').forEach(btn => {
            btn.addEventListener('click', () => {
                const kbId = btn.getAttribute('data-kb-id');
                this.switchKnowledgeBase(kbId);
            });
        });
    }

    switchKnowledgeBase(kbId) {
        const kb = this.knowledgeBases.find(k => k.id === kbId);
        if (!kb || !kb.available) return;

        this.currentKnowledgeBase = kb;
        
        // Update button states
        this.modeSelector.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.getAttribute('data-kb-id') === kbId);
        });
        
        // Update UI
        this.updateUI();
        
        // Load the chat history for this knowledge base
        this.loadChatHistory(kbId);
    }

    updateUI() {
        if (!this.currentKnowledgeBase) return;

        const isCodeReview = this.currentKnowledgeBase.type === 'code_review';
        
        // Show/hide panels
        this.reviewPanel.classList.toggle('hidden', !isCodeReview);
        
        // Update layout
        document.body.classList.toggle('vectorbt-mode', !isCodeReview);
        
        // Update chat title and status
        this.chatTitle.textContent = `💬 Chat - ${this.currentKnowledgeBase.name}`;
        this.kbStatus.textContent = this.currentKnowledgeBase.supports_images ? '📷 Images supportées' : '';
        
        // Show/hide image button based on support
        this.imageBtn.classList.toggle('hidden', !this.currentKnowledgeBase.supports_images);
    }

    loadChatHistory(kbId) {
        // Initialize history for this knowledge base if it doesn't exist
        if (!this.chatHistories[kbId]) {
            this.chatHistories[kbId] = [];
        }

        // Clear the chat display
        this.chatMessages.innerHTML = '';
        
        // Load existing messages for this knowledge base
        const history = this.chatHistories[kbId];
        if (history.length === 0) {
            // Add welcome message if no history exists
            this.addWelcomeMessage();
        } else {
            // Restore chat history
            history.forEach(message => {
                this.displayMessage(message.sender, message.content, message.images);
            });
        }
    }

    addWelcomeMessage() {
        if (!this.currentKnowledgeBase) return;
        
        let welcomeMessage;
        if (this.currentKnowledgeBase.type === 'code_review') {
            welcomeMessage = 'Bonjour ! Collez votre code et posez vos questions pour une review détaillée.';
        } else {
            welcomeMessage = `Bonjour ! Je peux vous aider avec ${this.currentKnowledgeBase.description.toLowerCase()}. Posez-moi vos questions !`;
        }
        
        if (this.currentKnowledgeBase.supports_images) {
            welcomeMessage += ' Vous pouvez aussi ajouter des images pour plus de contexte.';
        }
            
        this.addMessage('assistant', welcomeMessage);
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
        if (!this.currentKnowledgeBase) {
            this.sendBtn.disabled = true;
            return;
        }

        const hasQuestion = this.questionInput.value.trim().length > 0;
        const hasImages = this.selectedImages.length > 0;
        const isCodeReview = this.currentKnowledgeBase.type === 'code_review';
        const hasCode = !isCodeReview || this.codeEditor.value.trim().length > 0;
        
        // Allow sending if there's a question OR images (or both), and code if needed
        const canSend = (hasQuestion || hasImages) && hasCode;
        
        this.sendBtn.disabled = !canSend || this.isLoading;
    }

    handleImageSelect(event) {
        const files = Array.from(event.target.files);
        
        files.forEach(file => {
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const imageData = {
                        file: file,
                        dataUrl: e.target.result,
                        name: file.name
                    };
                    this.selectedImages.push(imageData);
                    this.updateImagePreview();
                    this.updateSendButton();
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Clear the input so the same file can be selected again
        event.target.value = '';
    }

    updateImagePreview() {
        if (this.selectedImages.length === 0) {
            this.imagePreviewContainer.classList.remove('has-images');
            this.imagePreviewContainer.innerHTML = '';
            return;
        }

        this.imagePreviewContainer.classList.add('has-images');
        this.imagePreviewContainer.innerHTML = this.selectedImages.map((image, index) => `
            <div class="image-preview">
                <img src="${image.dataUrl}" alt="${image.name}" title="${image.name}">
                <button class="remove-image" onclick="ragAssistant.removeImage(${index})" title="Supprimer l'image">×</button>
            </div>
        `).join('');
    }

    removeImage(index) {
        this.selectedImages.splice(index, 1);
        this.updateImagePreview();
        this.updateSendButton();
    }

    autoResizeTextarea(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }

    async sendMessage() {
        if (this.isLoading || this.sendBtn.disabled || !this.currentKnowledgeBase) return;

        const question = this.questionInput.value.trim();
        const hasImages = this.selectedImages.length > 0;
        
        if (!question && !hasImages) return;

        // Add user message to chat (with images if any)
        this.addMessage('user', question, this.selectedImages);
        
        // Clear input and images
        this.questionInput.value = '';
        const imagesToSend = [...this.selectedImages]; // Copy for API call
        this.selectedImages = [];
        this.updateImagePreview();
        this.updateSendButton();
        this.autoResizeTextarea(this.questionInput);

        // Show loading
        this.setLoading(true);

        try {
            let response;
            
            if (this.currentKnowledgeBase.type === 'code_review') {
                const code = this.codeEditor.value.trim();
                if (!code) {
                    throw new Error('Veuillez fournir du code à analyser');
                }
                response = await this.reviewCode(code, question, imagesToSend);
            } else {
                // Show loading overlay for database operations
                this.showLoadingOverlay(`Interrogation de ${this.currentKnowledgeBase.name}...`);
                response = await this.queryKnowledgeBase(this.currentKnowledgeBase.id, question, imagesToSend);
                this.hideLoadingOverlay();
            }

            // Add assistant response
            this.addMessage('assistant', response.response || response.message || 'Réponse reçue');
            
        } catch (error) {
            console.error('Erreur:', error);
            this.hideLoadingOverlay();
            
            // Check if it's a database building error
            if (error.message.includes('Failed to build') || error.message.includes('Building')) {
                this.addMessage('assistant', `🔄 Construction de la base de données en cours... Veuillez patienter quelques minutes et réessayer.`);
            } else {
                this.addMessage('assistant', `❌ Erreur: ${error.message}`);
            }
        } finally {
            this.setLoading(false);
        }
    }

    async queryKnowledgeBase(kbId, question, images = []) {
        const formData = new FormData();
        formData.append('question', question || '');
        
        images.forEach((image, index) => {
            formData.append(`image_${index}`, image.file);
        });

        const response = await fetch(`/query/${kbId}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Erreur lors de la requête');
        }

        return await response.json();
    }

    async reviewCode(code, question, images = []) {
        const formData = new FormData();
        formData.append('code', code);
        formData.append('question', question || '');
        
        images.forEach((image, index) => {
            formData.append(`image_${index}`, image.file);
        });

        const response = await fetch('/review/code', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Erreur lors de la review');
        }

        return await response.json();
    }

    addMessage(sender, content, images = []) {
        // Display the message
        this.displayMessage(sender, content, images);
        
        // Add to history for current knowledge base
        if (this.currentKnowledgeBase) {
            const kbId = this.currentKnowledgeBase.id;
            if (!this.chatHistories[kbId]) {
                this.chatHistories[kbId] = [];
            }
            this.chatHistories[kbId].push({ sender, content, images, timestamp: new Date() });
        }
    }

    displayMessage(sender, content, images = []) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${sender}`;
        
        const contentEl = document.createElement('div');
        contentEl.className = 'message-content';
        
        // Add images if any
        if (images && images.length > 0) {
            const imagesHtml = images.map(image => 
                `<img src="${image.dataUrl}" alt="${image.name}" style="max-width: 200px; max-height: 200px; border-radius: 8px; margin: 5px; display: block;">`
            ).join('');
            contentEl.innerHTML = imagesHtml;
        }
        
        // Add text content if any
        if (content && content.trim()) {
            const formattedContent = this.formatMessage(content);
            if (images && images.length > 0) {
                contentEl.innerHTML += `<div style="margin-top: 10px;">${formattedContent}</div>`;
            } else {
                contentEl.innerHTML = formattedContent;
            }
        }
        
        messageEl.appendChild(contentEl);
        this.chatMessages.appendChild(messageEl);

        // Enhance code blocks: ensure correct whitespace, add wrap toggle & copy, then optional highlighting
        setTimeout(() => {
            const preList = messageEl.querySelectorAll('pre');

            preList.forEach((pre, idx) => {
                // Preserve default no-wrap; users can toggle wrapping
                pre.classList.remove('is-wrapped');

                const codeEl = pre.querySelector('code');

                // Build optional header with language, wrap toggle, copy button (non-intrusive)
                const langClass = codeEl?.className || '';
                const lang = (langClass.match(/language-([a-z0-9_+-]+)/i) || [,'plain'])[1];

                // Only add header once
                if (!pre.previousElementSibling || !pre.previousElementSibling.classList.contains('code-header')) {
                    const header = document.createElement('div');
                    header.className = 'code-header';
                    header.innerHTML = `
                        <span class="code-lang">${lang}</span>
                        <div>
                            <button class="code-action-btn js-wrap-toggle" type="button" title="Toggle wrap">Wrap</button>
                            <button class="code-action-btn js-copy-code" type="button" title="Copy">Copy</button>
                        </div>
                    `;
                    pre.parentNode.insertBefore(header, pre);

                    // Wire wrap toggle
                    header.querySelector('.js-wrap-toggle').addEventListener('click', () => {
                        pre.classList.toggle('is-wrapped');
                    });

                    // Wire copy
                    header.querySelector('.js-copy-code').addEventListener('click', async () => {
                        try {
                            const text = codeEl ? codeEl.textContent : pre.textContent;
                            await navigator.clipboard.writeText(text);
                            const btn = header.querySelector('.js-copy-code');
                            const original = btn.textContent;
                            btn.textContent = 'Copied';
                            setTimeout(() => btn.textContent = original, 1500);
                        } catch (e) {
                            console.error('Copy failed', e);
                        }
                    });
                }
            });

            // With marked+Prism, content is already highlighted; keep a defensive re-run
            if (typeof Prism !== 'undefined' && Prism.highlightAllUnder) {
                Prism.highlightAllUnder(messageEl);
            }
        }, 50);

        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    formatMessage(content) {
        // Render Markdown using marked and rely on Prism for highlighting.
        // Keep post-render header/wrap/copy logic elsewhere (already in displayMessage).
        const src = (content ?? '').toString();

        // Configure marked once (idempotent)
        if (!this._markedConfigured) {
            if (typeof marked !== 'undefined') {
                // Safe defaults: escape HTML; enable GitHub-flavored markdown
                marked.setOptions({
                    gfm: true,
                    breaks: false,
                    headerIds: true,
                    mangle: false
                });
                // Sanitize by escaping raw HTML using a custom renderer hook
                const renderer = new marked.Renderer();

                // Let code blocks render with language class Prism expects
                renderer.code = (code, infostring) => {
                    const lang = (infostring || '').trim() || 'plain';
                    const escaped = this.escapeHtml(code.replace(/[ \t]+$/gm, ''));
                    return `<pre><code class="language-${lang}">${escaped}</code></pre>`;
                };

                // Inline code
                renderer.codespan = (code) => {
                    return `<code class="inline-code">${this.escapeHtml(code)}</code>`;
                };

                // Paragraphs
                renderer.paragraph = (text) => {
                    return `<p>${text}</p>`;
                };

                // Escape any raw HTML blocks to avoid injection since we are not enabling HTML
                renderer.html = (html) => {
                    return this.escapeHtml(html);
                };

                marked.use({ renderer });
            }
            this._markedConfigured = true;
        }

        if (typeof marked === 'undefined' || typeof marked.parse !== 'function') {
            // Fallback: plain escaped text
            return `<p>${this.escapeHtml(src)}</p>`;
        }

        // Primary render
        return marked.parse(src);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    copyCode(codeId) {
        const codeElement = document.getElementById(codeId);
        if (codeElement) {
            const text = codeElement.textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Show feedback
                const copyBtn = codeElement.closest('.code-block-container').querySelector('.copy-btn');
                const originalText = copyBtn.querySelector('.copy-text').textContent;
                copyBtn.querySelector('.copy-text').textContent = 'Copied!';
                copyBtn.classList.add('copied');
                
                setTimeout(() => {
                    copyBtn.querySelector('.copy-text').textContent = originalText;
                    copyBtn.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy code:', err);
            });
        }
    }

    async clearChat() {
        if (!this.currentKnowledgeBase) return;
        
        const kbId = this.currentKnowledgeBase.id;
        
        // Clear the display
        this.chatMessages.innerHTML = '';
        
        // Clear history for current knowledge base
        this.chatHistories[kbId] = [];
        
        // Clear images
        this.selectedImages = [];
        this.updateImagePreview();
        
        // Clear server-side history for this knowledge base
        if (this.currentKnowledgeBase.type !== 'code_review') {
            try {
                await fetch(`/clear-history/${kbId}`, {
                    method: 'POST'
                });
            } catch (error) {
                console.error('Failed to clear server history:', error);
            }
        }
        
        // Add welcome message
        this.addWelcomeMessage();
    }

    setLoading(loading) {
        this.isLoading = loading;
        if (loading) {
            this.sendBtn.classList.add('loading');
        } else {
            this.sendBtn.classList.remove('loading');
        }
        this.updateSendButton();
    }

    showLoadingOverlay(message) {
        // Remove existing overlay if any
        this.hideLoadingOverlay();
        
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-message">${message}</div>
                <div class="loading-submessage">Cela peut prendre quelques minutes...</div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Fade in
        setTimeout(() => overlay.classList.add('visible'), 10);
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
            setTimeout(() => {
                if (overlay.parentNode) {
                    overlay.parentNode.removeChild(overlay);
                }
            }, 300);
        }
    }
}

// Initialize the application when DOM is loaded
let ragAssistant;
document.addEventListener('DOMContentLoaded', () => {
    ragAssistant = new RAGAssistant();
});