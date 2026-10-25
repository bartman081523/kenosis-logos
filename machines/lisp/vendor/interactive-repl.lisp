;;;; interactive-repl.lisp
;;;; Interactive REPL for conversing with the philosopher

(format t "~%~%")
(format t "╔════════════════════════════════════════════════════════════╗~%")
(format t "║    Interactive Learning Philosopher                       ║~%")
(format t "║    Learns your language as you speak                      ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")
(format t "~%")

;; Load system
(format t "Loading system...~%")
(load "package.lisp")
(load "utils.lisp")
(load "goedel-encoder.lisp")
(load "symbolic-operator.lisp")
(load "geometric-tokenizer.lisp")
(load "cognito-construct.lisp")
(load "dialectic-macros.lisp")
(load "living-logos.lisp")
(load "reflection-engine.lisp")
(load "philosopher.lisp")
(load "letter-speech.lisp")
(load "persistence.lisp")
(load "language-learning.lisp")
(load "language-learning.lisp")
;; (load "translator.lisp") ; Not needed for data generation
(format t "✅ System loaded~%~%")

(in-package :lisp-cognito-philosopher)
(setf *log-level* :warn)

;; Initialize Deterministic RNG (Standard Seed = 42)
(set-rng-seed 42)
(format t "🎲 Deterministic RNG initialized (Seed 42)~%")

;; Create philosopher
(format t "Creating philosopher...~%")
(defparameter *interactive-philosopher* (make-philosopher ""))

;; Select checkpoint and load state
(format t "~%")
(let ((selected-checkpoint (select-checkpoint)))
  (if selected-checkpoint
      (progn
        (format t "~%Loading checkpoint: ~A~%" (file-namestring selected-checkpoint))
        (load-state-from-file *interactive-philosopher* selected-checkpoint))
      (format t "~%Starting fresh session.~%")))

;; Create new session file for this conversation
(format t "~%")
(create-session-file)

(format t "~%╔════════════════════════════════════════════════════════════╗~%")
(format t "║  How it works:                                            ║~%")
(format t "║                                                            ║~%")
(format t "║  1. Type your message and press Enter                     ║~%")
(format t "║  2. Philosopher learns from your words                    ║~%")
(format t "║  3. Philosopher responds in generated language            ║~%")
(format t "║  4. All saved to current session file                     ║~%")
(format t "║                                                            ║~%")
(format t "║  Commands:                                                ║~%")
(format t "║  - Type 'quit' or 'exit' to end                           ║~%")
(format t "║  - Type 'state' to see current state                      ║~%")
(format t "║  - Type 'gap' to see Gödel Gap                            ║~%")
(format t "╚════════════════════════════════════════════════════════════╝~%")
(format t "~%")

;; Main interaction loop
(defun run-interactive-session ()
  "Run interactive conversation session."
  (loop
    (format t "~%YOU: ")
    (force-output)
    
    (let ((input (read-line)))
      
      ;; Check for commands
      (cond
        ((or (string-equal input "quit") (string-equal input "exit"))
         (save-final-state *interactive-philosopher*)
         (format t "~%👋 Goodbye!~%")
         (return))
        
        ((string-equal input "state")
         (let ((state (cognito-construct-state *interactive-philosopher*)))
           (format t "STATE: ~{~,3F~^ ~}~%" (coerce state 'list)))
         (continue))
        
        ((string-equal input "gap")
         (format t "Gödel Gap: ~,3F~%" (get-goedel-gap *interactive-philosopher*))
         (continue))
        
        (t
         ;; Learn from input
         (learn-from-input *interactive-philosopher* input)
         
         ;; Generate response
         (let* ((response (generate-response *interactive-philosopher* input))
                (state (cognito-construct-state *interactive-philosopher*)))
           (format t "PHILOSOPHER: ~A~%" response)
           
           ;; Display translation - DISABLE FOR DATA GEN
           ;; (let ((translations (translate-response response state)))
           ;;   (when translations
           ;;     (format t "  ┌─ Translation:~%")
           ;;     (dolist (trans translations)
           ;;       (format t "  │ ~A → ~A~%"
           ;;               (getf trans :original)
           ;;               (getf trans :translation)))
           ;;     (format t "  └─────────────~%")))
           
           ;; Save interaction (not full state, just conversation)
           (save-interaction *interactive-philosopher* input response)))))))

;; Start session
(format t "Session started~%")
(format t "~%")
(run-interactive-session)

;; Exit
(sb-ext:exit :code 0)
