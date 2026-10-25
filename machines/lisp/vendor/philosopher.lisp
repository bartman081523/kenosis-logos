;;;; philosopher.lisp
;;;; Main entry point for the Lisp Cognito Philosopher

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; PHILOSOPHER CREATION
;;; ============================================================================

(defun make-philosopher (problem &key 
                                  (lexicon-path nil)
                                  (knowledge-data nil))
  "Create a new philosopher to contemplate a philosophical problem.
   
   problem: The philosophical question to explore
   lexicon-path: Optional path to geometric lexicon JSON file
   knowledge-data: Optional initial knowledge (hash table or alist)"
  (let ((cognito (make-cognito-construct :problem problem)))
    
    ;; Load geometric tokenizer if lexicon provided
    (when lexicon-path
      (let ((tokenizer (make-geometric-tokenizer)))
        (when (load-lexicon tokenizer lexicon-path)
          (setf (cognito-construct-tokenizer cognito) tokenizer))))
    
    ;; Initialize with knowledge
    (when knowledge-data
      (construct cognito knowledge-data))
    
    (log-message :info "🧙 Philosopher created to contemplate: ~A" problem)
    cognito))

;;; ============================================================================
;;; PHILOSOPHER EXECUTION
;;; ============================================================================

(defun run-philosopher (cognito &key 
                                 (queries '("consciousness" "existence" "meaning"))
                                 (n-cycles 10)
                                 (self-modify-threshold 0.5)
                                 (generate-text t))
  "Run the philosopher through multiple thought cycles.
   
   queries: List of queries to think about
   n-cycles: Number of thought cycles to run
   self-modify-threshold: Gödel Gap threshold for self-modification
   generate-text: Whether to generate philosophical text at the end
   
   Returns: A report of the philosophical journey"
  (log-message :info "🚀 Starting philosophical journey (~D cycles)..." n-cycles)
  
  (let ((report (make-hash-table :test 'equal))
        (constructs (when generate-text
                     (list (make-living-construct :emergence)
                           (make-living-construct :information)
                           (make-living-construct :consciousness)
                           (make-living-construct :transcendence)))))
    
    ;; Main thought loop
    (dotimes (cycle n-cycles)
      (log-message :info "~%--- Cycle ~D/~D ---" (1+ cycle) n-cycles)
      
      ;; Think about each query
      (dolist (query queries)
        (think cognito query))
      
      ;; Reflect on self
      (let ((reflection (reflect cognito)))
        (log-message :info "Gap: ~,3F (~A)" 
                    (getf reflection :gap)
                    (getf reflection :interpretation))
        
        ;; Self-modify if gap is high
        (when (> (getf reflection :gap) self-modify-threshold)
          (log-message :info "🔄 Self-modifying due to high gap...")
          (self-correct cognito)))
      
      ;; Output current understanding
      (when (zerop (mod cycle 5))
        (output cognito)))
    
    ;; Generate philosophical text if requested
    (when generate-text
      (log-message :info "~%📝 Generating philosophical text...")
      (let ((text (generate-philosophical-text constructs 
                                               '((x . 1.0))
                                               5)))
        (setf (gethash "philosophical-text" report) text)
        (format t "~A" text)))
    
    ;; Test consciousness criteria
    (log-message :info "~%🧪 Testing consciousness criteria...")
    (let ((criteria-results (check-consciousness-criteria cognito 50)))
      (setf (gethash "consciousness-criteria" report) criteria-results)
      (format t "~A" (report-consciousness-criteria criteria-results)))
    
    ;; Final output
    (log-message :info "~%📊 Final Report:")
    (let ((final-output (output cognito)))
      (setf (gethash "final-output" report) final-output))
    
    ;; Return report
    report))

;;; ============================================================================
;;; CONVENIENCE FUNCTIONS
;;; ============================================================================

(defun quick-philosopher (problem &optional (n-cycles 10))
  "Quick way to create and run a philosopher.
   Returns the philosopher instance."
  (let ((philosopher (make-philosopher problem)))
    (run-philosopher philosopher :n-cycles n-cycles)
    philosopher))

(defun demo-philosopher ()
  "Run a demonstration of the philosopher system."
  (format t "~%~%")
  (format t "╔════════════════════════════════════════════════════════════╗~%")
  (format t "║         Lisp Cognito Philosopher Demonstration            ║~%")
  (format t "╚════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  (let ((philosopher (make-philosopher "What is the nature of consciousness?"
                                       :knowledge-data '(("consciousness" . "self-reference")
                                                        ("existence" . "being")
                                                        ("meaning" . "purpose")))))
    (run-philosopher philosopher
                    :queries '("consciousness" "self-reference" "emergence")
                    :n-cycles 20
                    :self-modify-threshold 0.4
                    :generate-text t)
    
    (format t "~%~%")
    (format t "╔════════════════════════════════════════════════════════════╗~%")
    (format t "║                    Demonstration Complete                  ║~%")
    (format t "╚════════════════════════════════════════════════════════════╝~%")
    (format t "~%")
    
    philosopher))

;;; ============================================================================
;;; DIALECTIC DEMONSTRATION
;;; ============================================================================

(defun demo-dialectic ()
  "Demonstrate the dialectic macro system."
  (format t "~%~%")
  (format t "╔════════════════════════════════════════════════════════════╗~%")
  (format t "║           Dialectic System Demonstration                   ║~%")
  (format t "╚════════════════════════════════════════════════════════════╝~%")
  (format t "~%")
  
  ;; Define thesis
  (format t "Defining thesis...~%")
  (defthesis energy-thesis ()
    (generate-random-ast 2))
  
  ;; Define antithesis
  (format t "Defining antithesis...~%")
  (defantithesis energy-antithesis (thesis-result)
    (list 'negate thesis-result))
  
  ;; Define synthesis
  (format t "Defining synthesis...~%")
  (defsynthesis energy-synthesis energy-thesis energy-antithesis)
  
  ;; Run synthesis
  (format t "~%Running synthesis...~%")
  (let ((result (energy-synthesis)))
    (format t "Synthesis result: ~A~%" (ast-to-string result)))
  
  ;; Demonstrate self-modification
  (format t "~%Demonstrating self-modification...~%")
  (modify-generator "ENERGY-THESIS" (generate-random-ast 3))
  
  (format t "~%Running modified synthesis...~%")
  (let ((result (energy-synthesis)))
    (format t "Modified synthesis result: ~A~%" (ast-to-string result)))
  
  (format t "~%Dialectic demonstration complete.~%"))
