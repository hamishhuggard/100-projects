# Mini Projects Monorepo

I find that to best way to iterate code projects and learn new technical skills, is to to create a sequence of mini-projects. Messing around with git branches, and creating and organizing repos is annoying, so as a minimalist solution I have this monorepo where each mini-project is organized like this:
```bash
projects/
├── 001-example-project/
├── 002-example-project-v2/
└── 003-another-project/
```

A few interesting projects to take a look at:

### MLOps

- [DVC Demo](projects/176-dvc-demo)

### Deep learning

### React

### Agents and Chatbots


### Previews

I like to add screenshots to the README.md files, to show what they look like when they run. Raw screenshots are too large and get rejected by GitHub, so here's a script to downsize images to 1000px width:

```bash
python projects/resize-image/resize_image.py projects/n-some-project/preview.png
```

### Symlinks

To avoid duplicating images and other resources across projects, put them in the `/assets` directory at the root of this monorepo, and create symbolic links (symlink) within the projects:

```bash
ln -s projects/111-example-project/assets /assets
```

When the application server runs, it sees the symlink as a normal directory and can serve the files from it. GitHub will show the symlink as a normal directory, but doesn't actually store the duplicate files.

Apparently symlinks can be finnicky on Windows, so you may need to:
- Run your terminal as an administrator
- Enable developer mode in Windows
- Run the following command:
```bash
git config --global core.symlinks true
```
If those don't work, you may have to manually copy the assets into the project directories.