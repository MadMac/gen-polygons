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
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Padding},
    Terminal,
};
use tui_input::Input;
use tui_input::backend::crossterm::to_input_request;
use std::io::{self, stdout, Stdout};

/// Configuration for a new session
#[derive(Debug, Clone)]
pub struct NewSessionConfig {
    /// Session label/name
    pub label: String,
    /// Path to the goal image (default: "goal.png")
    pub goal: String,
    /// Run until Ctrl-C (default: false)
    pub infinite: bool,
    /// Open live window (default: false)
    pub show_window: bool,
}

impl Default for NewSessionConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            goal: "goal.png".to_string(),
            infinite: false,
            show_window: false,
        }
    }
}

/// Action to perform after TUI exits
#[derive(Debug, Clone)]
pub enum TuiAction {
    /// Run a new session with the given configuration
    NewSession(NewSessionConfig),
    /// Resume the session with the given ID
    ResumeSession { id: i64 },
    /// Delete the session with the given ID
    DeleteSession { id: i64 },
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
            return Ok(TuiAction::NewSession(NewSessionConfig::default()));
        }
    };

    // Create app state
    let mut app = App {
        sessions: sessions.clone(),
        list_state: ListState::default().with_selected(Some(0)),
        mode: Mode::SessionSelect,
        new_session_config: NewSessionConfig::default(),
        new_session_label_input: Input::new("".to_string()),
        new_session_goal_input: Input::new("goal.png".to_string()),
        new_session_active_field: ActiveField::Label,
        session_to_delete: None,
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
                        KeyCode::Char('d') => {
                            let selected = app.list_state.selected();
                            if selected.is_some() && selected.unwrap() < app.sessions.len() {
                                app.session_to_delete = Some(app.sessions[selected.unwrap()].id);
                                app.mode = Mode::ConfirmDelete;
                            }
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
                    // Handle field navigation first
                    match key.code {
                        KeyCode::Tab => {
                            app.new_session_active_field = app.new_session_active_field.next();
                        }
                        KeyCode::BackTab => {
                            app.new_session_active_field = app.new_session_active_field.prev();
                        }
                        KeyCode::Down => {
                            app.new_session_active_field = app.new_session_active_field.next();
                        }
                        KeyCode::Up => {
                            app.new_session_active_field = app.new_session_active_field.prev();
                        }
                        KeyCode::Char(' ') => {
                            // Toggle boolean fields
                            match app.new_session_active_field {
                                ActiveField::Infinite => {
                                    app.new_session_config.infinite = !app.new_session_config.infinite;
                                }
                                ActiveField::ShowWindow => {
                                    app.new_session_config.show_window = !app.new_session_config.show_window;
                                }
                                _ => {}
                            }
                        }
                        KeyCode::Enter => {
                            let label = app.new_session_label_input.value();
                            if !label.is_empty() {
                                // Sync the input values to the config
                                app.new_session_config.label = label.to_string();
                                app.new_session_config.goal = app.new_session_goal_input.value().to_string();
                                
                                cleanup_terminal(&mut terminal)?;
                                return Ok(TuiAction::NewSession(app.new_session_config.clone()));
                            }
                        }
                        KeyCode::Esc => {
                            app.mode = Mode::SessionSelect;
                            app.new_session_label_input = Input::new("".to_string());
                            app.new_session_goal_input = Input::new("goal.png".to_string());
                            app.new_session_config = NewSessionConfig::default();
                            app.new_session_active_field = ActiveField::Label;
                        }
                        _ => {
                            // Handle text input for active text fields using tui-input
                            if let Some(request) = to_input_request(&CrosstermEvent::Key(key)) {
                                match app.new_session_active_field {
                                    ActiveField::Label => {
                                        app.new_session_label_input.handle(request);
                                    }
                                    ActiveField::Goal => {
                                        app.new_session_goal_input.handle(request);
                                    }
                                    ActiveField::Infinite | ActiveField::ShowWindow => {}
                                }
                            }
                        }
                    }
                }
                Mode::ConfirmDelete => {
                    match key.code {
                        KeyCode::Char('y') | KeyCode::Enter => {
                            if let Some(id) = app.session_to_delete {
                                cleanup_terminal(&mut terminal)?;
                                return Ok(TuiAction::DeleteSession { id });
                            }
                        }
                        KeyCode::Char('n') | KeyCode::Esc => {
                            app.mode = Mode::SessionSelect;
                            app.session_to_delete = None;
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
    new_session_config: NewSessionConfig,
    new_session_label_input: Input,
    new_session_goal_input: Input,
    new_session_active_field: ActiveField,
    session_to_delete: Option<i64>,
}

/// TUI modes
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    SessionSelect,
    NewSession,
    ConfirmDelete,
}

/// Fields in the new session configuration screen
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveField {
    Label,
    Goal,
    Infinite,
    ShowWindow,
}

impl ActiveField {
    fn next(self) -> Self {
        match self {
            ActiveField::Label => ActiveField::Goal,
            ActiveField::Goal => ActiveField::Infinite,
            ActiveField::Infinite => ActiveField::ShowWindow,
            ActiveField::ShowWindow => ActiveField::Label,
        }
    }

    fn prev(self) -> Self {
        match self {
            ActiveField::Label => ActiveField::ShowWindow,
            ActiveField::Goal => ActiveField::Label,
            ActiveField::Infinite => ActiveField::Goal,
            ActiveField::ShowWindow => ActiveField::Infinite,
        }
    }
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
                    Span::raw("n: New  "),
                    Span::raw("d: Delete  "),
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
            // Create a centered block for new session configuration
            let block = Block::default()
                .title("New Session Configuration")
                .borders(Borders::ALL);

            // Use a simpler approach: fixed percentage that should work well
            // We need space for "Infinite Mode" and "Show Window" plus prefixes
            // Let's use 60% width and 20% height for a good balance
            let area = centered_rect(60, 20, size);
            f.render_widget(Clear, area);
            f.render_widget(block, area);

            // Calculate inner area with padding
            let inner_area = Rect {
                x: area.x + 1,
                y: area.y + 1,
                width: area.width.saturating_sub(2),
                height: area.height.saturating_sub(2),
            };

            // Field positioning - ensure each field has enough space
            let field_height = 1;
            let line_spacing = 1; // Add spacing between fields

            // Render label input with prefix
            let label_rect = Rect {
                x: inner_area.x,
                y: inner_area.y,
                width: inner_area.width,
                height: field_height,
            };
            
            let label_style = if app.new_session_active_field == ActiveField::Label {
                Style::new().fg(ratatui::style::Color::Yellow)
            } else {
                Style::new()
            };
            
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::styled("Label: ", label_style),
                ])),
                label_rect
            );
            // Render label input with cursor
            let label_text = app.new_session_label_input.value();
            let label_cursor = app.new_session_label_input.cursor();
            let label_display = render_input_field("Label: ", label_text, label_cursor, app.new_session_active_field == ActiveField::Label);
            f.render_widget(Paragraph::new(label_display), label_rect);

            // Render goal input with cursor
            let goal_rect = Rect {
                x: inner_area.x,
                y: inner_area.y + field_height + line_spacing,
                width: inner_area.width,
                height: field_height,
            };
            
            let goal_text = app.new_session_goal_input.value();
            let goal_cursor = app.new_session_goal_input.cursor();
            let goal_display = render_input_field("Goal: ", goal_text, goal_cursor, app.new_session_active_field == ActiveField::Goal);
            f.render_widget(Paragraph::new(goal_display), goal_rect);

            // Render boolean fields
            let bool_style = |is_active: bool| {
                if is_active {
                    Style::new().fg(ratatui::style::Color::Yellow).underlined()
                } else {
                    Style::new()
                }
            };

            let infinite_rect = Rect {
                x: inner_area.x,
                y: inner_area.y + (field_height + line_spacing) * 2,
                width: inner_area.width,
                height: field_height,
            };
            let checkbox = if app.new_session_config.infinite { "[✓]" } else { "[ ]" };
            let cursor = if app.new_session_active_field == ActiveField::Infinite { "▋" } else { "" };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::raw(checkbox),
                    Span::raw(" "),
                    Span::styled("Infinite Mode", bool_style(app.new_session_active_field == ActiveField::Infinite)),
                    Span::raw(cursor),
                ])),
                infinite_rect
            );

            let window_rect = Rect {
                x: inner_area.x,
                y: inner_area.y + (field_height + line_spacing) * 3,
                width: inner_area.width,
                height: field_height,
            };
            let checkbox = if app.new_session_config.show_window { "[✓]" } else { "[ ]" };
            let cursor = if app.new_session_active_field == ActiveField::ShowWindow { "▋" } else { "" };
            f.render_widget(
                Paragraph::new(Line::from(vec![
                    Span::raw(checkbox),
                    Span::raw(" "),
                    Span::styled("Show Window", bool_style(app.new_session_active_field == ActiveField::ShowWindow)),
                    Span::raw(cursor),
                ])),
                window_rect
            );

            // Instructions at the bottom
            let instructions_rect = Rect {
                x: inner_area.x,
                y: inner_area.y + (field_height + line_spacing) * 4,
                width: inner_area.width,
                height: 1,
            };
            let instructions = Paragraph::new(vec![
                Line::from(vec![
                    Span::raw("↑/↓:"),
                    Span::styled("Navigate", Style::new().fg(ratatui::style::Color::Green)),
                    Span::raw(" | Tab:"),
                    Span::styled("Next", Style::new().fg(ratatui::style::Color::Green)),
                    Span::raw(" | Shift+Tab:"),
                    Span::styled("Prev", Style::new().fg(ratatui::style::Color::Green)),
                    Span::raw(" | Space:"),
                    Span::styled("Toggle", Style::new().fg(ratatui::style::Color::Green)),
                    Span::raw(" | Enter:"),
                    Span::styled("Create", Style::new().fg(ratatui::style::Color::Green)),
                    Span::raw(" | Esc:"),
                    Span::styled("Cancel", Style::new().fg(ratatui::style::Color::Red)),
                ]),
            ]);
            f.render_widget(instructions, instructions_rect);
        }
        Mode::ConfirmDelete => {
            // Create a centered confirmation dialog
            let block = Block::default()
                .title("Confirm Delete")
                .borders(Borders::ALL)
                .padding(Padding::new(2, 2, 2, 2));

            let area = centered_rect(50, 8, size);
            f.render_widget(Clear, area);
            f.render_widget(block, area);

            // Confirmation message
            if let Some(id) = app.session_to_delete {
                let message = Paragraph::new(vec![
                    Line::from(vec![
                        Span::raw(format!("Delete session {}?", id)),
                    ]),
                    Line::from(vec![
                        Span::raw(""),
                    ]),
                    Line::from(vec![
                        Span::styled("This cannot be undone!", Style::new().fg(ratatui::style::Color::Red)),
                    ]),
                ])
                .block(Block::default());

                let inner_area = Rect {
                    x: area.x + 2,
                    y: area.y + 2,
                    width: area.width - 4,
                    height: 4,
                };
                f.render_widget(message, inner_area);

                // Instructions
                let instructions = Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled("y: Yes", Style::new().fg(ratatui::style::Color::Green)),
                        Span::raw(" | "),
                        Span::styled("n: No", Style::new().fg(ratatui::style::Color::Red)),
                    ]),
                ])
                .block(Block::default());

                f.render_widget(instructions, Rect {
                    x: area.x + 2,
                    y: area.y + area.height - 2,
                    width: area.width - 4,
                    height: 1,
                });
            }
        }
    }
}

/// Helper function to render an input field with cursor at the right position
fn render_input_field<'a>(prefix: &'a str, value: &'a str, cursor_pos: usize, is_active: bool) -> Line<'a> {
    let prefix_style = if is_active {
        Style::new().fg(ratatui::style::Color::Yellow).underlined()
    } else {
        Style::new()
    };
    
    let mut spans = vec![
        Span::styled(prefix, prefix_style),
    ];
    
    // Add the value text with cursor
    let chars: Vec<char> = value.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        spans.push(Span::raw(c.to_string()));
        // Insert cursor after this character if we're at the cursor position
        if is_active && i + 1 == cursor_pos {
            spans.push(Span::raw("▋").fg(ratatui::style::Color::Yellow));
        }
    }
    
    // If cursor is at the end, add it after all content
    if is_active && cursor_pos == chars.len() {
        spans.push(Span::raw("▋").fg(ratatui::style::Color::Yellow));
    }
    
    Line::from(spans)
}

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_new_session_config_default() {
            let config = NewSessionConfig::default();
            
            assert_eq!(config.label, "");
            assert_eq!(config.goal, "goal.png");
            assert!(!config.infinite);
            assert!(!config.show_window);
        }

        #[test]
        fn test_active_field_navigation() {
            assert_eq!(ActiveField::Label.next(), ActiveField::Goal);
            assert_eq!(ActiveField::Goal.next(), ActiveField::Infinite);
            assert_eq!(ActiveField::Infinite.next(), ActiveField::ShowWindow);
            assert_eq!(ActiveField::ShowWindow.next(), ActiveField::Label);

            assert_eq!(ActiveField::Label.prev(), ActiveField::ShowWindow);
            assert_eq!(ActiveField::Goal.prev(), ActiveField::Label);
            assert_eq!(ActiveField::Infinite.prev(), ActiveField::Goal);
            assert_eq!(ActiveField::ShowWindow.prev(), ActiveField::Infinite);
        }

        #[test]
        fn test_new_session_config_with_custom_values() {
            let config = NewSessionConfig {
                label: "My Session".to_string(),
                goal: "custom_goal.png".to_string(),
                infinite: true,
                show_window: true,
            };
            
            assert_eq!(config.label, "My Session");
            assert_eq!(config.goal, "custom_goal.png");
            assert!(config.infinite);
            assert!(config.show_window);
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
