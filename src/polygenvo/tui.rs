//! Ratatui TUI for session management.
//!
//! This module provides a terminal-based UI for:
//! - Listing and selecting saved sessions
//! - Creating new sessions with custom parameters
//! - Resuming previous sessions

use crate::persistence::{list_sessions, SessionSummary};
use crossterm::{
    event::{self, Event as CrosstermEvent, KeyCode, KeyEventKind},
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Padding, Wrap},
    Terminal,
};
use std::io::{self, stdout, Stdout};

/// Action to perform after TUI exits
#[derive(Debug, Clone)]
pub enum TuiAction {
    /// Run a new session with the given label
    NewSession { label: String },
    /// Resume the session with the given ID
    ResumeSession { id: i64 },
    /// Exit without running any session
    Exit,
}

/// Run the TUI session selection screen.
/// Returns the action the user selected.
pub fn run_tui(db_conn: &mut rusqlite::Connection) -> io::Result<TuiAction> {
    // Setup terminal
    let mut stdout = stdout();
    enable_raw_mode()?;
    crossterm::execute!(
        stdout,
        EnterAlternateScreen,
    )?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Load sessions
    let sessions = match list_sessions(db_conn) {
        Ok(s) => s,
        Err(_) => {
            cleanup_terminal(&mut terminal)?;
            return Ok(TuiAction::NewSession { label: "default".to_string() });
        }
    };

    // Create app state
    let mut app = App {
        sessions: sessions.clone(),
        list_state: ListState::default().with_selected(Some(0)),
        mode: Mode::SessionSelect,
        new_session_label: String::new(),
    };

    // Main loop
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if let CrosstermEvent::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match app.mode {
                Mode::SessionSelect => {
                    match key.code {
                        KeyCode::Up => {
                            app.list_state.select_previous();
                        }
                        KeyCode::Down => {
                            app.list_state.select_next();
                        }
                        KeyCode::Char('n') => {
                            app.mode = Mode::NewSession;
                        }
                        KeyCode::Enter => {
                            let selected = app.list_state.selected();
                            if selected.is_some() && selected.unwrap() < app.sessions.len() {
                                cleanup_terminal(&mut terminal)?;
                                return Ok(TuiAction::ResumeSession {
                                    id: app.sessions[selected.unwrap()].id,
                                });
                            } else {
                                app.mode = Mode::NewSession;
                            }
                        }
                        KeyCode::Esc | KeyCode::Char('q') => {
                            cleanup_terminal(&mut terminal)?;
                            return Ok(TuiAction::Exit);
                        }
                        _ => {}
                    }
                }
                Mode::NewSession => {
                    match key.code {
                        KeyCode::Char(c) => {
                            app.new_session_label.push(c);
                        }
                        KeyCode::Backspace => {
                            app.new_session_label.pop();
                        }
                        KeyCode::Enter => {
                            if !app.new_session_label.is_empty() {
                                cleanup_terminal(&mut terminal)?;
                                return Ok(TuiAction::NewSession {
                                    label: app.new_session_label.clone(),
                                });
                            }
                        }
                        KeyCode::Esc => {
                            app.mode = Mode::SessionSelect;
                            app.new_session_label.clear();
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Cleanup terminal state
fn cleanup_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;
    Ok(())
}

/// App state
struct App {
    sessions: Vec<SessionSummary>,
    list_state: ListState,
    mode: Mode,
    new_session_label: String,
}

/// TUI modes
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    SessionSelect,
    NewSession,
}

/// UI rendering
fn ui(f: &mut ratatui::prelude::Frame, app: &mut App) {
    let size = f.size();

    match app.mode {
        Mode::SessionSelect => {
            // Create list items
            let mut items: Vec<ListItem> = app
                .sessions
                .iter()
                .map(|s| {
                    let label = s.label.as_deref().unwrap_or("<unnamed>");
                    let fitness_pct = (s.current_fitness as f64 / 1_000_000.0) * 100.0;
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("ID: {} | {}", s.id, label),
                            Style::new().bold(),
                        ),
                        Span::raw(format!(
                            " | Fitness: {:.2}% | {} tris | Phase: {} | Steps: {}",
                            fitness_pct, s.triangle_count, s.phase_index, s.steps_run
                        )),
                    ]))
                })
                .collect();

            // Add "New Session" option
            items.push(ListItem::new(Line::from(vec![
                Span::styled("[NEW SESSION]", Style::new().bold().fg(ratatui::style::Color::Green)),
            ])));

            let list = List::new(items)
                .block(
                    Block::default()
                        .title("Select Session")
                        .borders(Borders::ALL),
                )
                .highlight_style(Style::new().bg(ratatui::style::Color::Blue).fg(ratatui::style::Color::White))
                .highlight_symbol("> ");

            f.render_stateful_widget(list, size, &mut app.list_state);

            // Show instructions
            let instructions = Paragraph::new(vec![
                Line::from(vec![
                    Span::raw("↑/↓: Navigate  "),
                    Span::raw("Enter: Select  "),
                    Span::raw("n: New Session  "),
                    Span::raw("q: Quit"),
                ]),
            ])
            .block(Block::default());

            f.render_widget(instructions, Rect {
                x: 0,
                y: size.height - 1,
                width: size.width,
                height: 1,
            });
        }
        Mode::NewSession => {
            // Create a centered block for new session input
            let block = Block::default()
                .title("New Session")
                .borders(Borders::ALL)
                .padding(Padding::new(2, 2, 2, 2));

            let area = centered_rect(60, 10, size);
            f.render_widget(Clear, area);
            f.render_widget(block, area);

            // Input field
            let input = Paragraph::new(app.new_session_label.as_str())
                .block(
                    Block::default()
                        .title("Session Label")
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: false });

            let inner_area = Rect {
                x: area.x + 2,
                y: area.y + 2,
                width: area.width - 4,
                height: 3,
            };
            f.render_widget(input, inner_area);

            // Instructions
            let instructions = Paragraph::new(vec![
                Line::from(vec![
                    Span::raw("Enter: Create  "),
                    Span::raw("Esc: Cancel"),
                ]),
            ])
            .block(Block::default());

            f.render_widget(instructions, Rect {
                x: area.x,
                y: area.y + area.height - 1,
                width: area.width,
                height: 1,
            });
        }
    }
}

/// Helper to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
