# Semantic versioning

Semantic Versioning uses a three-part number system in the format MAJOR.MINOR.PATCH.

MAJOR version should be incremented when there are incompatible API changes. For example, if your library or software was at version 1.4.3 and you made changes that would break or disrupt the functionality of someone using your software, you would update the version to 2.0.0.

MINOR version should be incremented when you add functionality in a backwards-compatible manner. So if you add a new feature but everything else remains the same, you'd increment this number. For example, from 1.4.3 to 1.5.0.

PATCH version should be incremented when you make backwards-compatible bug fixes. This is what would change when going from 0.0.1 to 0.0.2. For instance, you fixed a bug, but there were no other changes to the code that would disrupt the user.

# Push new releases to PyPI

Create a new tag with the version number and push it to GitHub.
GitHub Actions will automatically build and publish the package to PyPI.

```bash
git tag -a v0.0.1 -m "Release version 0.0.1"
git push origin v0.0.1
```
