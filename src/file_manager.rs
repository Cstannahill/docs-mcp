use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs as async_fs;
use chrono::{DateTime, Utc};
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManager {
    pub base_directory: PathBuf,
    pub allowed_extensions: Vec<String>,
    pub max_file_size: u64, // in bytes
    pub read_only_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
    pub is_directory: bool,
    pub is_file: bool,
    pub extension: Option<String>,
    pub created: Option<DateTime<Utc>>,
    pub modified: Option<DateTime<Utc>>,
    pub permissions: FilePermissions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePermissions {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryListing {
    pub path: PathBuf,
    pub files: Vec<FileInfo>,
    pub directories: Vec<FileInfo>,
    pub total_files: usize,
    pub total_directories: usize,
    pub total_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: PathBuf,
    pub matches: Vec<SearchMatch>,
    pub total_matches: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMatch {
    pub line_number: usize,
    pub line_content: String,
    pub column_start: usize,
    pub column_end: usize,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}

impl FileManager {
    pub fn new(base_directory: PathBuf) -> Self {
        Self {
            base_directory,
            allowed_extensions: vec![
                "txt".to_string(), "md".to_string(), "rs".to_string(),
                "js".to_string(), "ts".to_string(), "py".to_string(),
                "json".to_string(), "yaml".to_string(), "yml".to_string(),
                "toml".to_string(), "cfg".to_string(), "conf".to_string(),
                "html".to_string(), "css".to_string(), "xml".to_string(),
                "log".to_string(), "csv".to_string(), "sql".to_string(),
            ],
            max_file_size: 10 * 1024 * 1024, // 10MB
            read_only_mode: false,
        }
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only_mode = read_only;
        self
    }

    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }

    pub fn with_allowed_extensions(mut self, extensions: Vec<String>) -> Self {
        self.allowed_extensions = extensions;
        self
    }

    /// Read file contents with safety checks
    pub async fn read_file(&self, file_path: &str) -> Result<String> {
        let path = self.resolve_path(file_path)?;
        self.validate_read_access(&path)?;

        let metadata = async_fs::metadata(&path).await?;
        if metadata.len() > self.max_file_size {
            return Err(anyhow::anyhow!("File too large: {} bytes (max: {})", metadata.len(), self.max_file_size));
        }

        let content = async_fs::read_to_string(&path).await?;
        Ok(content)
    }

    /// Read file with line range
    pub async fn read_file_lines(&self, file_path: &str, start_line: Option<usize>, end_line: Option<usize>) -> Result<String> {
        let content = self.read_file(file_path).await?;
        let lines: Vec<&str> = content.lines().collect();
        
        let start = start_line.unwrap_or(1).saturating_sub(1);
        let end = end_line.unwrap_or(lines.len()).min(lines.len());
        
        if start >= lines.len() {
            return Ok(String::new());
        }
        
        let selected_lines = &lines[start..end];
        Ok(selected_lines.join("\n"))
    }

    /// Write file contents with safety checks
    pub async fn write_file(&self, file_path: &str, content: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let path = self.resolve_path(file_path)?;
        self.validate_write_access(&path)?;

        if content.len() as u64 > self.max_file_size {
            return Err(anyhow::anyhow!("Content too large: {} bytes (max: {})", content.len(), self.max_file_size));
        }

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            async_fs::create_dir_all(parent).await?;
        }

        async_fs::write(&path, content).await?;
        Ok(())
    }

    /// Append to file
    pub async fn append_file(&self, file_path: &str, content: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let path = self.resolve_path(file_path)?;
        self.validate_write_access(&path)?;

        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        tokio::io::AsyncWriteExt::write_all(&mut file, content.as_bytes()).await?;
        Ok(())
    }

    /// Delete file
    pub async fn delete_file(&self, file_path: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let path = self.resolve_path(file_path)?;
        self.validate_write_access(&path)?;

        if path.is_file() {
            async_fs::remove_file(&path).await?;
        } else if path.is_dir() {
            async_fs::remove_dir_all(&path).await?;
        } else {
            return Err(anyhow::anyhow!("Path does not exist: {}", path.display()));
        }

        Ok(())
    }

    /// Create directory
    pub async fn create_directory(&self, dir_path: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let path = self.resolve_path(dir_path)?;
        self.validate_write_access(&path)?;

        async_fs::create_dir_all(&path).await?;
        Ok(())
    }

    /// List directory contents
    pub async fn list_directory(&self, dir_path: &str) -> Result<DirectoryListing> {
        let path = self.resolve_path(dir_path)?;
        self.validate_read_access(&path)?;

        let mut files = Vec::new();
        let mut directories = Vec::new();
        let mut total_size = 0u64;

        let mut entries = async_fs::read_dir(&path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let metadata = entry.metadata().await?;
            
            let file_info = FileInfo {
                path: entry_path.clone(),
                name: entry_path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                size: metadata.len(),
                is_directory: metadata.is_dir(),
                is_file: metadata.is_file(),
                extension: entry_path.extension()
                    .map(|s| s.to_string_lossy().to_string()),
                created: metadata.created().ok()
                    .map(|t| DateTime::from(t)),
                modified: metadata.modified().ok()
                    .map(|t| DateTime::from(t)),
                permissions: FilePermissions {
                    readable: metadata.permissions().readonly() == false,
                    writable: !metadata.permissions().readonly(),
                    executable: false, // Simplified for now
                },
            };

            total_size += metadata.len();

            if metadata.is_dir() {
                directories.push(file_info);
            } else {
                files.push(file_info);
            }
        }

        // Sort by name
        files.sort_by(|a, b| a.name.cmp(&b.name));
        directories.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(DirectoryListing {
            path,
            total_files: files.len(),
            total_directories: directories.len(),
            total_size,
            files,
            directories,
        })
    }

    /// Search for files by name pattern
    pub async fn find_files(&self, pattern: &str, dir_path: Option<&str>) -> Result<Vec<FileInfo>> {
        let search_path = if let Some(dir) = dir_path {
            self.resolve_path(dir)?
        } else {
            self.base_directory.clone()
        };

        let regex = Regex::new(pattern)?;
        let mut results = Vec::new();

        self.find_files_recursive(&search_path, &regex, &mut results).await?;
        Ok(results)
    }

    /// Search for text within files
    pub async fn search_in_files(&self, pattern: &str, dir_path: Option<&str>, context_lines: usize) -> Result<Vec<SearchResult>> {
        let search_path = if let Some(dir) = dir_path {
            self.resolve_path(dir)?
        } else {
            self.base_directory.clone()
        };

        let regex = Regex::new(pattern)?;
        let mut results = Vec::new();

        self.search_in_files_recursive(&search_path, &regex, context_lines, &mut results).await?;
        Ok(results)
    }

    /// Get file information
    pub async fn get_file_info(&self, file_path: &str) -> Result<FileInfo> {
        let path = self.resolve_path(file_path)?;
        self.validate_read_access(&path)?;

        let metadata = async_fs::metadata(&path).await?;
        
        Ok(FileInfo {
            path: path.clone(),
            name: path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: metadata.len(),
            is_directory: metadata.is_dir(),
            is_file: metadata.is_file(),
            extension: path.extension()
                .map(|s| s.to_string_lossy().to_string()),
            created: metadata.created().ok()
                .map(|t| DateTime::from(t)),
            modified: metadata.modified().ok()
                .map(|t| DateTime::from(t)),
            permissions: FilePermissions {
                readable: true, // Simplified
                writable: !metadata.permissions().readonly(),
                executable: false,
            },
        })
    }

    /// Copy file
    pub async fn copy_file(&self, source: &str, destination: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let src_path = self.resolve_path(source)?;
        let dst_path = self.resolve_path(destination)?;
        
        self.validate_read_access(&src_path)?;
        self.validate_write_access(&dst_path)?;

        if let Some(parent) = dst_path.parent() {
            async_fs::create_dir_all(parent).await?;
        }

        async_fs::copy(&src_path, &dst_path).await?;
        Ok(())
    }

    /// Move/rename file
    pub async fn move_file(&self, source: &str, destination: &str) -> Result<()> {
        if self.read_only_mode {
            return Err(anyhow::anyhow!("File manager is in read-only mode"));
        }

        let src_path = self.resolve_path(source)?;
        let dst_path = self.resolve_path(destination)?;
        
        self.validate_read_access(&src_path)?;
        self.validate_write_access(&dst_path)?;

        if let Some(parent) = dst_path.parent() {
            async_fs::create_dir_all(parent).await?;
        }

        async_fs::rename(&src_path, &dst_path).await?;
        Ok(())
    }

    // Helper methods
    fn resolve_path(&self, file_path: &str) -> Result<PathBuf> {
        let path = if Path::new(file_path).is_absolute() {
            PathBuf::from(file_path)
        } else {
            self.base_directory.join(file_path)
        };

        // Normalize path and check it's within base directory
        let canonical_base = self.base_directory.canonicalize()
            .unwrap_or_else(|_| self.base_directory.clone());
        
        let canonical_path = if path.exists() {
            path.canonicalize().unwrap_or(path)
        } else {
            path
        };

        if !canonical_path.starts_with(&canonical_base) {
            return Err(anyhow::anyhow!("Path outside allowed directory: {}", canonical_path.display()));
        }

        Ok(canonical_path)
    }

    fn validate_read_access(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(anyhow::anyhow!("File or directory does not exist: {}", path.display()));
        }
        Ok(())
    }

    fn validate_write_access(&self, path: &Path) -> Result<()> {
        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy().to_lowercase();
            if !self.allowed_extensions.contains(&ext) {
                return Err(anyhow::anyhow!("File extension not allowed: {}", ext));
            }
        }
        Ok(())
    }

    async fn find_files_recursive(&self, dir: &Path, regex: &Regex, results: &mut Vec<FileInfo>) -> Result<()> {
        let mut entries = async_fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let metadata = entry.metadata().await?;
            
            if metadata.is_dir() {
                // Recurse into subdirectory
                if let Err(_) = Box::pin(self.find_files_recursive(&path, regex, results)).await {
                    // Skip directories we can't read
                    continue;
                }
            } else if metadata.is_file() {
                if let Some(file_name) = path.file_name() {
                    let name = file_name.to_string_lossy();
                    if regex.is_match(&name) {
                        let file_info = FileInfo {
                            path: path.clone(),
                            name: name.to_string(),
                            size: metadata.len(),
                            is_directory: false,
                            is_file: true,
                            extension: path.extension().map(|s| s.to_string_lossy().to_string()),
                            created: metadata.created().ok().map(|t| DateTime::from(t)),
                            modified: metadata.modified().ok().map(|t| DateTime::from(t)),
                            permissions: FilePermissions {
                                readable: true,
                                writable: !metadata.permissions().readonly(),
                                executable: false,
                            },
                        };
                        results.push(file_info);
                    }
                }
            }
        }
        
        Ok(())
    }

    async fn search_in_files_recursive(&self, dir: &Path, regex: &Regex, context_lines: usize, results: &mut Vec<SearchResult>) -> Result<()> {
        let mut entries = async_fs::read_dir(dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            let metadata = entry.metadata().await?;
            
            if metadata.is_dir() {
                if let Err(_) = Box::pin(self.search_in_files_recursive(&path, regex, context_lines, results)).await {
                    continue;
                }
            } else if metadata.is_file() {
                // Check if file extension is allowed for searching
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if !self.allowed_extensions.contains(&ext_str) {
                        continue;
                    }
                }

                // Skip large files
                if metadata.len() > self.max_file_size {
                    continue;
                }

                if let Ok(content) = async_fs::read_to_string(&path).await {
                    let matches = self.find_matches_in_content(&content, regex, context_lines);
                    if !matches.is_empty() {
                        results.push(SearchResult {
                            file_path: path,
                            total_matches: matches.len(),
                            matches,
                        });
                    }
                }
            }
        }
        
        Ok(())
    }

    fn find_matches_in_content(&self, content: &str, regex: &Regex, context_lines: usize) -> Vec<SearchMatch> {
        let lines: Vec<&str> = content.lines().collect();
        let mut matches = Vec::new();

        for (line_index, line) in lines.iter().enumerate() {
            for mat in regex.find_iter(line) {
                let context_start = line_index.saturating_sub(context_lines);
                let context_end = (line_index + context_lines + 1).min(lines.len());

                let context_before = if line_index > context_start {
                    lines[context_start..line_index].iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                };

                let context_after = if line_index + 1 < context_end {
                    lines[line_index + 1..context_end].iter().map(|s| s.to_string()).collect()
                } else {
                    Vec::new()
                };

                matches.push(SearchMatch {
                    line_number: line_index + 1,
                    line_content: line.to_string(),
                    column_start: mat.start(),
                    column_end: mat.end(),
                    context_before,
                    context_after,
                });
            }
        }

        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_manager = FileManager::new(temp_dir.path().to_path_buf());

        // Test write and read
        let content = "Hello, world!";
        file_manager.write_file("test.txt", content).await.unwrap();
        let read_content = file_manager.read_file("test.txt").await.unwrap();
        assert_eq!(content, read_content);
    }

    #[tokio::test]
    async fn test_directory_listing() {
        let temp_dir = TempDir::new().unwrap();
        let file_manager = FileManager::new(temp_dir.path().to_path_buf());

        file_manager.write_file("file1.txt", "content1").await.unwrap();
        file_manager.write_file("file2.txt", "content2").await.unwrap();
        file_manager.create_directory("subdir").await.unwrap();

        let listing = file_manager.list_directory(".").await.unwrap();
        assert_eq!(listing.total_files, 2);
        assert_eq!(listing.total_directories, 1);
    }
}
