//! Persistence layer for saving and restoring ES state to SQLite database.
//!
//! This module provides functionality to:
//! - Save complete evolution state (genome, fitness, parameters) to SQLite
//! - Load saved sessions and resume evolution
//! - List all saved sessions
//! - Delete sessions
//!
//! The database is stored in `triangles/checkpoints.db` alongside snapshot images.

use std::path::Path;

use crate::genome::Vertex;
use thiserror::Error;

/// Schema version for future migrations
const SCHEMA_VERSION: u32 = 1;

/// SQL for creating database tables
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    label TEXT,
    goal_width INTEGER NOT NULL, goal_height INTEGER NOT NULL,
    goal_image_data BLOB NOT NULL,
    phase_index INTEGER NOT NULL, phase_step INTEGER NOT NULL,
    current_fitness INTEGER NOT NULL, initial_fitness INTEGER NOT NULL,
    steps_run INTEGER NOT NULL, improvements_total INTEGER NOT NULL,
    sigma_pos REAL NOT NULL, sigma_col REAL NOT NULL, sigma_grad REAL NOT NULL,
    window_steps INTEGER NOT NULL,
    pos_gen INTEGER NOT NULL, pos_better INTEGER NOT NULL,
    col_gen INTEGER NOT NULL, col_better INTEGER NOT NULL,
    grad_gen INTEGER NOT NULL, grad_better INTEGER NOT NULL,
    schedule_accepts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS genome_vertices (
    session_id INTEGER NOT NULL, idx INTEGER NOT NULL,
    position_x REAL NOT NULL, position_y REAL NOT NULL, position_z REAL NOT NULL,
    color_r REAL NOT NULL, color_g REAL NOT NULL, color_b REAL NOT NULL, color_a REAL NOT NULL,
    PRIMARY KEY (session_id, idx),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
"#;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during persistence operations.
#[derive(Error, Debug)]
pub enum PersistenceError {
    /// SQLite database error
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    /// Requested session was not found
    #[error("Session {0} not found")]
    SessionNotFound(i64),

    /// Checkpoint data is corrupted
    #[error("Corrupted checkpoint data: {0}")]
    CorruptedData(String),

    /// Checkpoint data failed validation
    #[error("Invalid checkpoint: {0}")]
    InvalidCheckpoint(String),

    /// Database schema version mismatch
    #[error("Schema version mismatch: expected {0}, got {1}")]
    SchemaVersionMismatch(u32, u32),
}

// ============================================================================
// Data Structures
// ============================================================================

/// Complete state of an ES run at a point in time.
/// All fields must be serializable to/from SQLite.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    // Identity
    pub session_id: Option<i64>,
    pub label: Option<String>,
    
    // Goal image (raw RGBA8 bytes, row-major)
    pub goal_width: u32,
    pub goal_height: u32,
    pub goal_pixels: Vec<u8>,
    
    // ES State
    pub current_genome: Vec<Vertex>,
    pub current_fitness: i64,  // i64 for 32-bit platform safety
    pub initial_fitness: i64,
    pub steps_run: u64,
    pub improvements_total: u64,
    
    // Phase Schedule
    pub phase_idx: usize,
    pub phase_step: u64,
    pub schedule_accepts: u64,
    
    // Step Sizes
    pub sigma_pos: f32,
    pub sigma_col: f32,
    pub sigma_grad: f32,
    
    // 1/5 Rule Window
    pub window_steps: u64,
    pub pos_gen: u64,
    pub pos_better: u64,
    pub col_gen: u64,
    pub col_better: u64,
    pub grad_gen: u64,
    pub grad_better: u64,
}

impl Checkpoint {
    /// Create a new checkpoint with the current timestamp
    pub fn new(label: Option<String>) -> Self {
        Self {
            session_id: None,
            label,
            goal_width: 0,
            goal_height: 0,
            goal_pixels: Vec::new(),
            current_genome: Vec::new(),
            current_fitness: 0,
            initial_fitness: 0,
            steps_run: 0,
            improvements_total: 0,
            phase_idx: 0,
            phase_step: 0,
            schedule_accepts: 0,
            sigma_pos: 0.0,
            sigma_col: 0.0,
            sigma_grad: 0.0,
            window_steps: 0,
            pos_gen: 0,
            pos_better: 0,
            col_gen: 0,
            col_better: 0,
            grad_gen: 0,
            grad_better: 0,
        }
    }
}

/// Summary information for listing sessions (doesn't load full genome)
#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub id: i64,
    pub label: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub goal_width: u32,
    pub goal_height: u32,
    pub current_fitness: i64,
    pub triangle_count: usize,
    pub phase_index: usize,
    pub steps_run: u64,
}

// ============================================================================
// Database Initialization
// ============================================================================

/// Open or create the database at the given path.
/// Enables WAL mode for better concurrency and sets appropriate pragmas.
pub fn init_db(path: &Path) -> Result<rusqlite::Connection, PersistenceError> {
    let conn = rusqlite::Connection::open_with_flags(
        path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            | rusqlite::OpenFlags::SQLITE_OPEN_URI,
    )?;

    // Enable WAL mode for better concurrency (readers don't block writers)
    conn.pragma_update(None, "journal_mode", "WAL")?;
    
    // Synchronous = NORMAL for better performance
    // (we can tolerate some data loss on crash during occasional saves)
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    
    // Enable foreign key constraints
    conn.pragma_update(None, "foreign_keys", "ON")?;

    // Initialize schema
    conn.execute_batch(SCHEMA_SQL)?;

    Ok(conn)
}

/// Check and validate schema version.
fn check_schema_version(conn: &rusqlite::Connection) -> Result<(), PersistenceError> {
    let version: u32 = conn
        .query_row(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .map_err(|e| {
            // If table doesn't exist, we have a fresh DB which is fine
            if e.to_string().contains("no such table") {
                PersistenceError::Database(e)
            } else {
                PersistenceError::Database(e)
            }
        })?;

    if version != SCHEMA_VERSION {
        return Err(PersistenceError::SchemaVersionMismatch(
            SCHEMA_VERSION,
            version,
        ));
    }

    Ok(())
}

// ============================================================================
// Validation
// ============================================================================

/// Validate a checkpoint before saving or after loading.
/// Returns error if any field is invalid.
pub fn validate_checkpoint(checkpoint: &Checkpoint) -> Result<(), PersistenceError> {
    // Genome must have 3N vertices (N triangles) - empty is allowed for new sessions
    if checkpoint.current_genome.len() % 3 != 0 && !checkpoint.current_genome.is_empty() {
        return Err(PersistenceError::InvalidCheckpoint(format!(
            "genome has {} vertices (not multiple of 3)",
            checkpoint.current_genome.len()
        )));
    }

    // Goal dimensions must be positive
    if checkpoint.goal_width == 0 || checkpoint.goal_height == 0 {
        return Err(PersistenceError::InvalidCheckpoint(
            "goal dimensions must be positive".into(),
        ));
    }

    // Goal pixel data size must match dimensions
    let expected_size = (checkpoint.goal_width as usize)
        .checked_mul(checkpoint.goal_height as usize)
        .ok_or_else(|| {
            PersistenceError::InvalidCheckpoint(
                "goal dimensions overflow usize".into(),
            )
        })?
        .checked_mul(4)
        .ok_or_else(|| {
            PersistenceError::InvalidCheckpoint(
                "goal pixel count overflow usize".into(),
            )
        })?;

    if checkpoint.goal_pixels.len() != expected_size {
        return Err(PersistenceError::InvalidCheckpoint(format!(
            "goal_pixels length {} != expected {} for {}x{} RGBA8",
            checkpoint.goal_pixels.len(),
            expected_size,
            checkpoint.goal_width,
            checkpoint.goal_height
        )));
    }

    // All vertex positions and colors must be finite (not NaN or Inf)
    for (i, v) in checkpoint.current_genome.iter().enumerate() {
        for (j, &pos) in v.position.iter().enumerate() {
            if !pos.is_finite() {
                return Err(PersistenceError::CorruptedData(format!(
                    "vertex {} position[{}] is not finite: {}",
                    i, j, pos
                )));
            }
        }
        for (j, &color) in v.color.iter().enumerate() {
            if !color.is_finite() {
                return Err(PersistenceError::CorruptedData(format!(
                    "vertex {} color[{}] is not finite: {}",
                    i, j, color
                )));
            }
        }
    }

    // Fitness values must be non-negative and reasonable
    if checkpoint.current_fitness < 0 || checkpoint.initial_fitness < 0 {
        return Err(PersistenceError::InvalidCheckpoint(
            "fitness values must be non-negative".into(),
        ));
    }

    // Steps and counts must be non-negative
    if checkpoint.steps_run < checkpoint.improvements_total {
        return Err(PersistenceError::InvalidCheckpoint(
            "steps_run must be >= improvements_total".into(),
        ));
    }

    Ok(())
}

// ============================================================================
// Save Operations
// ============================================================================

/// Save a checkpoint to the database.
/// Returns the session ID (new or existing).
/// Uses a transaction for atomicity - either all data is saved or none.
pub fn save_session(
    conn: &mut rusqlite::Connection,
    checkpoint: &Checkpoint,
) -> Result<i64, PersistenceError> {
    // Validate before saving
    validate_checkpoint(checkpoint)?;

    let tx = conn.transaction()?;

    let now = chrono::Local::now().to_rfc3339();

    // Determine if this is an update or insert
    let session_id = if let Some(id) = checkpoint.session_id {
        // Update existing session
        tx.execute(
            "UPDATE sessions SET 
                updated_at = ?,
                label = ?,
                goal_width = ?,
                goal_height = ?,
                goal_image_data = ?,
                phase_index = ?,
                phase_step = ?,
                current_fitness = ?,
                initial_fitness = ?,
                steps_run = ?,
                improvements_total = ?,
                sigma_pos = ?,
                sigma_col = ?,
                sigma_grad = ?,
                window_steps = ?,
                pos_gen = ?,
                pos_better = ?,
                col_gen = ?,
                col_better = ?,
                grad_gen = ?,
                grad_better = ?,
                schedule_accepts = ?
             WHERE id = ?",
            rusqlite::params![
                now,
                checkpoint.label,
                checkpoint.goal_width as i64,
                checkpoint.goal_height as i64,
                &checkpoint.goal_pixels,
                checkpoint.phase_idx as i64,
                checkpoint.phase_step as i64,
                checkpoint.current_fitness,
                checkpoint.initial_fitness,
                checkpoint.steps_run as i64,
                checkpoint.improvements_total as i64,
                checkpoint.sigma_pos,
                checkpoint.sigma_col,
                checkpoint.sigma_grad,
                checkpoint.window_steps as i64,
                checkpoint.pos_gen as i64,
                checkpoint.pos_better as i64,
                checkpoint.col_gen as i64,
                checkpoint.col_better as i64,
                checkpoint.grad_gen as i64,
                checkpoint.grad_better as i64,
                checkpoint.schedule_accepts as i64,
                id,
            ],
        )?;
        id
    } else {
        // Insert new session
        let id: i64 = tx.query_row(
            "INSERT INTO sessions (
                created_at, updated_at, label,
                goal_width, goal_height, goal_image_data,
                phase_index, phase_step,
                current_fitness, initial_fitness,
                steps_run, improvements_total,
                sigma_pos, sigma_col, sigma_grad,
                window_steps, pos_gen, pos_better,
                col_gen, col_better, grad_gen, grad_better,
                schedule_accepts
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)
             RETURNING id",
            rusqlite::params![
                now,
                now,
                checkpoint.label,
                checkpoint.goal_width as i64,
                checkpoint.goal_height as i64,
                &checkpoint.goal_pixels,
                checkpoint.phase_idx as i64,
                checkpoint.phase_step as i64,
                checkpoint.current_fitness,
                checkpoint.initial_fitness,
                checkpoint.steps_run as i64,
                checkpoint.improvements_total as i64,
                checkpoint.sigma_pos,
                checkpoint.sigma_col,
                checkpoint.sigma_grad,
                checkpoint.window_steps as i64,
                checkpoint.pos_gen as i64,
                checkpoint.pos_better as i64,
                checkpoint.col_gen as i64,
                checkpoint.col_better as i64,
                checkpoint.grad_gen as i64,
                checkpoint.grad_better as i64,
                checkpoint.schedule_accepts as i64,
            ],
            |row| row.get(0),
        )?;
        id
    };

    // Delete old vertices for this session (if updating)
    tx.execute(
        "DELETE FROM genome_vertices WHERE session_id = ?",
        [session_id],
    )?;

    // Insert all vertices
    {
        let mut stmt = tx.prepare(
            "INSERT INTO genome_vertices (
                session_id, idx,
                position_x, position_y, position_z,
                color_r, color_g, color_b, color_a
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        )?;

        for (idx, vertex) in checkpoint.current_genome.iter().enumerate() {
            stmt.execute(rusqlite::params![
                session_id,
                idx as i64,
                vertex.position[0],
                vertex.position[1],
                vertex.position[2],
                vertex.color[0],
                vertex.color[1],
                vertex.color[2],
                vertex.color[3],
            ])?;
        }
        // stmt drops here, releasing the borrow on tx
    }

    tx.commit()?;
    Ok(session_id)
}

// ============================================================================
// Load Operations
// ============================================================================

/// Load a checkpoint from the database by session ID.
/// Validates the loaded data before returning.
pub fn load_session(
    conn: &rusqlite::Connection,
    session_id: i64,
) -> Result<Checkpoint, PersistenceError> {
    // Load session metadata
    let mut stmt = conn.prepare_cached(
        "SELECT 
            id, label,
            goal_width, goal_height, goal_image_data,
            phase_index, phase_step,
            current_fitness, initial_fitness,
            steps_run, improvements_total,
            sigma_pos, sigma_col, sigma_grad,
            window_steps, pos_gen, pos_better,
            col_gen, col_better, grad_gen, grad_better,
            schedule_accepts
         FROM sessions
         WHERE id = ?",
    )?;

    let mut rows = stmt.query([session_id])?;
    let row = rows.next()
        .transpose()
        .ok_or(PersistenceError::SessionNotFound(session_id))??;

    let mut checkpoint = Checkpoint {
        session_id: Some(row.get(0)?),
        label: row.get(1)?,
        goal_width: row.get::<_, i64>(2)? as u32,
        goal_height: row.get::<_, i64>(3)? as u32,
        goal_pixels: row.get(4)?,
        phase_idx: row.get::<_, i64>(5)? as usize,
        phase_step: row.get::<_, i64>(6)? as u64,
        current_fitness: row.get(7)?,
        initial_fitness: row.get(8)?,
        steps_run: row.get::<_, i64>(9)? as u64,
        improvements_total: row.get::<_, i64>(10)? as u64,
        sigma_pos: row.get(11)?,
        sigma_col: row.get(12)?,
        sigma_grad: row.get(13)?,
        window_steps: row.get::<_, i64>(14)? as u64,
        pos_gen: row.get::<_, i64>(15)? as u64,
        pos_better: row.get::<_, i64>(16)? as u64,
        col_gen: row.get::<_, i64>(17)? as u64,
        col_better: row.get::<_, i64>(18)? as u64,
        grad_gen: row.get::<_, i64>(19)? as u64,
        grad_better: row.get::<_, i64>(20)? as u64,
        schedule_accepts: row.get::<_, i64>(21)? as u64,
        current_genome: Vec::new(),
    };

    // Load genome vertices
    let mut stmt = conn.prepare_cached(
        "SELECT idx, position_x, position_y, position_z,
                color_r, color_g, color_b, color_a
         FROM genome_vertices
         WHERE session_id = ?
         ORDER BY idx",
    )?;

    let mut rows = stmt.query([session_id])?;
    let mut vertices = Vec::with_capacity(checkpoint.current_genome.capacity());
    while let Some(row) = rows.next()? {
        vertices.push(Vertex {
            position: [
                row.get::<_, f32>(1)?,
                row.get::<_, f32>(2)?,
                row.get::<_, f32>(3)?,
            ],
            color: [
                row.get::<_, f32>(4)?,
                row.get::<_, f32>(5)?,
                row.get::<_, f32>(6)?,
                row.get::<_, f32>(7)?,
            ],
        });
    }
    checkpoint.current_genome = vertices;

    // Validate the loaded checkpoint
    validate_checkpoint(&checkpoint)?;

    Ok(checkpoint)
}

/// Load a session summary for listing (doesn't load full genome).
fn load_session_summary(_conn: &rusqlite::Connection, row: &rusqlite::Row) -> Result<SessionSummary, PersistenceError> {
    Ok(SessionSummary {
        id: row.get(0)?,
        label: row.get(1)?,
        created_at: row.get(2)?,
        updated_at: row.get(3)?,
        goal_width: row.get(4)?,
        goal_height: row.get(5)?,
        current_fitness: row.get(6)?,
        triangle_count: {
            let n_vertices: i64 = row.get(7)?;
            (n_vertices / 3) as usize
        },
        phase_index: row.get(8)?,
        steps_run: row.get(9)?,
    })
}

/// List all saved sessions, ordered by updated_at descending (most recent first).
pub fn list_sessions(conn: &rusqlite::Connection) -> Result<Vec<SessionSummary>, PersistenceError> {
    let mut stmt = conn.prepare_cached(
        "SELECT 
            s.id, s.label, s.created_at, s.updated_at,
            s.goal_width, s.goal_height,
            s.current_fitness,
            (SELECT COUNT(*) FROM genome_vertices gv WHERE gv.session_id = s.id) as n_vertices,
            s.phase_index, s.steps_run
         FROM sessions s
         ORDER BY s.updated_at DESC",
    )?;

    let mut rows = stmt.query([])?;
    let mut sessions = Vec::new();
    while let Some(row) = rows.next()? {
        let session = load_session_summary(conn, &row)?;
        sessions.push(session);
    }
    Ok(sessions)
}

/// Delete a session and all its associated data (genome vertices).
pub fn delete_session(conn: &mut rusqlite::Connection, session_id: i64) -> Result<(), PersistenceError> {
    let tx = conn.transaction()?;
    
    // Foreign key CASCADE will delete genome_vertices automatically
    let rows_affected = tx.execute("DELETE FROM sessions WHERE id = ?", [session_id])?;
    
    if rows_affected == 0 {
        return Err(PersistenceError::SessionNotFound(session_id));
    }
    
    tx.commit()?;
    Ok(())
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the default database path.
pub fn default_db_path() -> std::path::PathBuf {
    Path::new("triangles").join("checkpoints.db")
}

/// Initialize the default database, creating parent directory if needed.
pub fn init_default_db() -> Result<rusqlite::Connection, PersistenceError> {
    let path = default_db_path();
    std::fs::create_dir_all(path.parent().unwrap())
        .map_err(|e| PersistenceError::InvalidCheckpoint(e.to_string()))?;
    init_db(&path)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_checkpoint() -> Checkpoint {
        Checkpoint {
            session_id: None,
            label: Some("test session".to_string()),
            goal_width: 64,
            goal_height: 64,
            goal_pixels: vec![0u8; 64 * 64 * 4],
            current_genome: vec![
                Vertex {
                    position: [0.0, 0.0, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                Vertex {
                    position: [1.0, 0.0, 0.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                Vertex {
                    position: [0.0, 1.0, 0.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                },
            ],
            current_fitness: 500000,
            initial_fitness: 100000,
            steps_run: 100,
            improvements_total: 50,
            phase_idx: 0,
            phase_step: 50,
            schedule_accepts: 25,
            sigma_pos: 0.1,
            sigma_col: 0.1,
            sigma_grad: 0.1,
            window_steps: 10,
            pos_gen: 5,
            pos_better: 2,
            col_gen: 3,
            col_better: 1,
            grad_gen: 2,
            grad_better: 1,
        }
    }

    #[test]
    fn test_validate_valid_checkpoint() {
        let cp = create_test_checkpoint();
        assert!(validate_checkpoint(&cp).is_ok());
    }

    #[test]
    fn test_validate_invalid_genome_size() {
        let mut cp = create_test_checkpoint();
        cp.current_genome = vec![
            Vertex {
                position: [0.0; 3],
                color: [0.0; 4],
            },
            Vertex {
                position: [0.0; 3],
                color: [0.0; 4],
            },
        ]; // 2 vertices, not multiple of 3

        assert!(validate_checkpoint(&cp).is_err());
    }

    #[test]
    fn test_validate_zero_goal_dimension() {
        let mut cp = create_test_checkpoint();
        cp.goal_width = 0;

        assert!(validate_checkpoint(&cp).is_err());
    }

    #[test]
    fn test_validate_goal_pixel_size_mismatch() {
        let mut cp = create_test_checkpoint();
        cp.goal_pixels = vec![0u8; 100]; // Wrong size

        assert!(validate_checkpoint(&cp).is_err());
    }

    #[test]
    fn test_validate_nan_vertex() {
        let mut cp = create_test_checkpoint();
        cp.current_genome[0].position[0] = f32::NAN;

        assert!(validate_checkpoint(&cp).is_err());
    }

    #[test]
    fn test_validate_negative_fitness() {
        let mut cp = create_test_checkpoint();
        cp.current_fitness = -1;

        assert!(validate_checkpoint(&cp).is_err());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let mut conn = init_db(&db_path).unwrap();

        let cp = create_test_checkpoint();
        let id = save_session(&mut conn, &cp).unwrap();

        let loaded = load_session(&conn, id).unwrap();

        assert_eq!(loaded.session_id, Some(id));
        assert_eq!(loaded.label, cp.label);
        assert_eq!(loaded.goal_width, cp.goal_width);
        assert_eq!(loaded.goal_height, cp.goal_height);
        assert_eq!(loaded.goal_pixels, cp.goal_pixels);
        assert_eq!(loaded.current_genome, cp.current_genome);
        assert_eq!(loaded.current_fitness, cp.current_fitness);
        assert_eq!(loaded.initial_fitness, cp.initial_fitness);
        assert_eq!(loaded.steps_run, cp.steps_run);
        assert_eq!(loaded.improvements_total, cp.improvements_total);
        assert_eq!(loaded.phase_idx, cp.phase_idx);
        assert_eq!(loaded.phase_step, cp.phase_step);
        assert_eq!(loaded.schedule_accepts, cp.schedule_accepts);
        assert!((loaded.sigma_pos - cp.sigma_pos).abs() < f32::EPSILON);
        assert!((loaded.sigma_col - cp.sigma_col).abs() < f32::EPSILON);
        assert!((loaded.sigma_grad - cp.sigma_grad).abs() < f32::EPSILON);
        assert_eq!(loaded.window_steps, cp.window_steps);
        assert_eq!(loaded.pos_gen, cp.pos_gen);
        assert_eq!(loaded.pos_better, cp.pos_better);
        assert_eq!(loaded.col_gen, cp.col_gen);
        assert_eq!(loaded.col_better, cp.col_better);
        assert_eq!(loaded.grad_gen, cp.grad_gen);
        assert_eq!(loaded.grad_better, cp.grad_better);
    }

    #[test]
    fn test_list_sessions_empty() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let conn = init_db(&db_path).unwrap();

        let sessions = list_sessions(&conn).unwrap();
        assert!(sessions.is_empty());
    }

    #[test]
    fn test_list_sessions_ordered() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let mut conn = init_db(&db_path).unwrap();

        // Save multiple sessions
        for i in 0..3 {
            let mut cp = create_test_checkpoint();
            cp.label = Some(format!("session {}", i));
            save_session(&mut conn, &cp).unwrap();
            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let sessions = list_sessions(&conn).unwrap();
        assert_eq!(sessions.len(), 3);

        // Most recent should be first
        assert!(sessions[0].label.as_deref() == Some("session 2"));
        assert!(sessions[1].label.as_deref() == Some("session 1"));
        assert!(sessions[2].label.as_deref() == Some("session 0"));
    }

    #[test]
    fn test_delete_session() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let mut conn = init_db(&db_path).unwrap();

        let cp = create_test_checkpoint();
        let id = save_session(&mut conn, &cp).unwrap();

        // Verify it exists
        assert!(load_session(&conn, id).is_ok());

        // Delete it
        delete_session(&mut conn, id).unwrap();

        // Verify it's gone
        assert!(load_session(&conn, id).is_err());
    }

    #[test]
    fn test_delete_nonexistent_session() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let mut conn = init_db(&db_path).unwrap();

        let result = delete_session(&mut conn, 999);
        assert!(matches!(result, Err(PersistenceError::SessionNotFound(999))));
    }

    #[test]
    fn test_load_nonexistent_session() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let conn = init_db(&db_path).unwrap();

        let result = load_session(&conn, 999);
        assert!(matches!(result, Err(PersistenceError::SessionNotFound(999))));
    }
}
