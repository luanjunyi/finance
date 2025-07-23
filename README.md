Create a virtual environment (optional)

```bash
virtualenv myenv
```

Install pip-compile with (Optional)

```bash
pip install pip-tools
```

Compile the `requirements.in`

```bash
pip-compile requirements.in
```

Install all the dependencies in `requirements.txt`

```bash
pip install -r requirements.txt
```
