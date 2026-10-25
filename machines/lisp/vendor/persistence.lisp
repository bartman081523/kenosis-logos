;;;; persistence.lisp
;;;; Save and load philosopher state with session-based files

(in-package #:lisp-cognito-philosopher)

;;; ============================================================================
;;; SESSION MANAGEMENT
;;; ============================================================================

(defparameter *session-file* nil
  "Current session file (set at program start)")

(defparameter *session-directory* "sessions/"
  "Directory to store session files")

(defun get-timestamp ()
  "Get current timestamp as string."
  (multiple-value-bind (sec min hour day month year)
      (get-decoded-time)
    (format nil "~4,'0D-~2,'0D-~2,'0D_~2,'0D-~2,'0D-~2,'0D"
            year month day hour min sec)))

(defun ensure-session-directory ()
  "Create sessions directory if it doesn't exist."
  (ensure-directories-exist *session-directory*))

(defun list-session-files ()
  "List all session files sorted by timestamp (newest first)."
  (ensure-session-directory)
  (let ((files (directory (concatenate 'string *session-directory* "session_*.txt"))))
    (sort files #'string> :key #'namestring)))

(defun create-session-file ()
  "Create a new session file with current timestamp."
  (ensure-session-directory)
  (let ((filename (format nil "~Asession_~A.txt" 
                         *session-directory*
                         (get-timestamp))))
    (setf *session-file* filename)
    (with-open-file (out filename
                         :direction :output
                         :if-exists :supersede
                         :if-does-not-exist :create)
      (format out "=== SESSION STARTED: ~A ===~%~%" (get-timestamp)))
    (format t "📝 New session file: ~A~%" filename)
    filename))

(defun select-checkpoint ()
  "Let user select a checkpoint to load from.
   Returns the selected filename or NIL for new session."
  (let ((sessions (list-session-files)))
    (if (null sessions)
        (progn
          (format t "~%No previous sessions found. Starting fresh.~%")
          nil)
        (progn
          (format t "~%╔════════════════════════════════════════════════════════════╗~%")
          (format t "║  Available Checkpoints                                    ║~%")
          (format t "╚════════════════════════════════════════════════════════════╝~%~%")
          (format t "0. Start new session~%")
          (loop for session in sessions
                for i from 1
                do (format t "~D. ~A~%" i (file-namestring session)))
          (format t "~%Select checkpoint (0-~D): " (length sessions))
          (force-output)
          
          (let ((choice (read-line)))
            (handler-case
                (let ((num (parse-integer choice)))
                  (cond
                    ((= num 0) nil)
                    ((and (>= num 1) (<= num (length sessions)))
                     (nth (1- num) sessions))
                    (t (format t "Invalid choice. Starting new session.~%")
                       nil)))
              (error () 
                (format t "Invalid input. Starting new session.~%")
                nil)))))))

;;; ============================================================================
;;; STATE PERSISTENCE
;;; ============================================================================

(defun save-interaction (cognito user-input philosopher-response)
  "Append interaction with COMPLETE state to current session file."
  (when *session-file*
    (with-open-file (out *session-file*
                         :direction :output
                         :if-exists :append)
      (let ((state (cognito-construct-state cognito))
            (self-model (cognito-construct-self-model cognito))
            (gap (get-goedel-gap cognito))
            (thought-count (cognito-construct-thought-count cognito)))
        (format out "USER: ~A~%" user-input)
        (format out "PHILOSOPHER: ~A~%" philosopher-response)
        (format out "STATE: ~{~,6F~^ ~}~%" (coerce state 'list))
        (format out "SELF-MODEL: ~{~,6F~^ ~}~%" (coerce self-model 'list))
        (format out "GAP: ~,6F~%" gap)
        (format out "THOUGHT-COUNT: ~D~%~%" thought-count)))))

(defun load-state-from-file (cognito filename)
  "Load and replay ALL states from a session file for complete learning."
  (when (probe-file filename)
    (with-open-file (in filename :direction :input)
      (let ((all-states nil)
            (all-self-models nil)
            (all-user-inputs nil)
            (current-user nil)
            (state-count 0))
        
        ;; Read entire file and collect ALL states
        (loop for line = (read-line in nil nil)
              while line
              do (cond
                   ((search "USER: " line)
                    (setf current-user (subseq line 6)))
                   ((search "STATE: " line)
                    (push (parse-vector-line line "STATE: ") all-states)
                    (when current-user
                      (push current-user all-user-inputs)
                      (setf current-user nil)))
                   ((search "SELF-MODEL: " line)
                    (push (parse-vector-line line "SELF-MODEL: ") all-self-models))))
        
        ;; Reverse to get chronological order
        (setf all-states (nreverse all-states))
        (setf all-self-models (nreverse all-self-models))
        (setf all-user-inputs (nreverse all-user-inputs))
        (setf state-count (length all-states))
        
        ;; Replay ALL states to re-learn the entire conversation
        (when (> state-count 0)
          (format t "📚 Replaying ~D states from conversation history...~%" state-count)
          
          ;; Apply each state sequentially
          (loop for state across (coerce all-states 'vector)
                for self-model across (coerce all-self-models 'vector)
                for i from 1
                do (progn
                     (setf (cognito-construct-state cognito) state)
                     (setf (cognito-construct-self-model cognito) self-model)
                     (setf (cognito-construct-thought-count cognito) i)))
          
          (let ((final-gap (get-goedel-gap cognito)))
            (format t "✅ Replayed entire conversation from ~A~%" (file-namestring filename))
            (format t "   States processed: ~D~%" state-count)
            (format t "   Final gap: ~,3F~%" final-gap)
            (format t "   Thought count: ~D~%" state-count)
            t))))))

(defun save-final-state (cognito)
  "Save final state snapshot at end of session."
  (when *session-file*
    (with-open-file (out *session-file*
                         :direction :output
                         :if-exists :append)
      (let ((state (cognito-construct-state cognito))
            (self-model (cognito-construct-self-model cognito))
            (gap (get-goedel-gap cognito))
            (thought-count (cognito-construct-thought-count cognito)))
        
        (format out "~%=== FINAL STATE ===~%")
        (format out "STATE: ~{~,6F~^ ~}~%" (coerce state 'list))
        (format out "SELF-MODEL: ~{~,6F~^ ~}~%" (coerce self-model 'list))
        (format out "GAP: ~,6F~%" gap)
        (format out "THOUGHT-COUNT: ~D~%" thought-count)
        (format out "=== SESSION ENDED: ~A ===~%" (get-timestamp))))
    (format t "💾 Final state saved to ~A~%" (file-namestring *session-file*))))

;;; ============================================================================
;;; HELPER FUNCTIONS
;;; ============================================================================

(defun parse-vector-line (line prefix)
  "Parse a line like 'STATE: 0.1 0.2 0.3' into a vector."
  (let* ((data-str (subseq line (length prefix)))
         (numbers (mapcar #'(lambda (s) (coerce (read-from-string s) 'double-float))
                         (remove-if #'(lambda (s) (string= s ""))
                                   (split-string data-str #\Space)))))
    (make-array (length numbers) 
                :initial-contents numbers
                :element-type 'double-float)))

(defun parse-float-line (line prefix)
  "Parse a line like 'GAP: 0.5' into a float."
  (read-from-string (subseq line (length prefix))))

(defun parse-integer-line (line prefix)
  "Parse a line like 'THOUGHT-COUNT: 42' into an integer."
  (parse-integer (subseq line (length prefix))))

(defun split-string (string delimiter)
  "Split string by delimiter."
  (loop with result = nil
        with start = 0
        for pos = (position delimiter string :start start)
        do (push (subseq string start pos) result)
           (if pos
               (setf start (1+ pos))
               (return (nreverse result)))
        finally (return (nreverse result))))
