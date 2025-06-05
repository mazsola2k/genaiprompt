Howto use llama-cpp-python @ Windows 11 natively:

Pre-req:
Fix Build Failure for llama-cpp-python on Windows 11
🔧 Step 1: Ensure Visual Studio Build Tools are Properly Installed
Make sure you’ve installed:

✅ MSVC v143 - VS 2022 C++ x64/x86 build tools

✅ C++ CMake tools for Windows

✅ Windows 10 SDK

Using with CPU only:

 D:\Dev>pip install --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121/
Looking in indexes: https://pypi.org/simple, https://abetlen.github.io/llama-cpp-python/whl/cu121/
Collecting llama-cpp-python
  Downloading llama_cpp_python-0.3.9.tar.gz (67.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.9/67.9 MB 56.2 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Collecting typing-extensions>=4.5.0 (from llama-cpp-python)
  Downloading typing_extensions-4.14.0-py3-none-any.whl.metadata (3.0 kB)
Collecting numpy>=1.20.0 (from llama-cpp-python)
  Downloading numpy-2.2.6-cp312-cp312-win_amd64.whl.metadata (60 kB)
Collecting diskcache>=5.6.1 (from llama-cpp-python)
  Downloading diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)
Collecting jinja2>=2.11.3 (from llama-cpp-python)
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting MarkupSafe>=2.0 (from jinja2>=2.11.3->llama-cpp-python)
  Downloading MarkupSafe-3.0.2-cp312-cp312-win_amd64.whl.metadata (4.1 kB)
Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading numpy-2.2.6-cp312-cp312-win_amd64.whl (12.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.6/12.6 MB 52.8 MB/s eta 0:00:00
Downloading typing_extensions-4.14.0-py3-none-any.whl (43 kB)
Downloading MarkupSafe-3.0.2-cp312-cp312-win_amd64.whl (15 kB)
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml) ... done
  Created wheel for llama-cpp-python: filename=llama_cpp_python-0.3.9-cp312-cp312-win_amd64.whl size=3367822 sha256=ed85fe288f83bb07d179b352682772478655f493c0d0fc394c45a3b72d9dd4f2
  Stored in directory: C:\Users\Peter\AppData\Local\Temp\pip-ephem-wheel-cache-67zaf9fy\wheels\e9\22\42\98dca29f6195951fae2aa548582827a45306350e282ab30617
Successfully built llama-cpp-python
Installing collected packages: typing-extensions, numpy, MarkupSafe, diskcache, jinja2, llama-cpp-python
Successfully installed MarkupSafe-3.0.2 diskcache-5.6.3 jinja2-3.1.6 llama-cpp-python-0.3.9 numpy-2.2.6 typing-extensions-4.14.0

LLama 2 and LLama 4 are two models from the Llama series of large-scale multilingual models developed by Meta.

The main differences between Llama 2 and Llama 4 are:

1. **The number of trained languages**:
   Llama 2 is trained with 2 languages, whereas Llama 4 is trained with 4 languages.

2. **The model size**:
   Llama 2 has a small model size, whereas Llama 4 has a large model size.



Using with gpu:

pip install llama-cpp-python --force-reinstall --no-cache-dir --prefer-binary --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2
Looking in indexes: https://pypi.org/simple, https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2
Collecting llama-cpp-python
  Downloading llama_cpp_python-0.3.9.tar.gz (67.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.9/67.9 MB 37.0 MB/s eta 0:00:00
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Installing backend dependencies ... done
  Preparing metadata (pyproject.toml) ... done
Collecting typing-extensions>=4.5.0 (from llama-cpp-python)
  Downloading typing_extensions-4.14.0-py3-none-any.whl.metadata (3.0 kB)
Collecting numpy>=1.20.0 (from llama-cpp-python)
  Downloading numpy-2.2.6-cp312-cp312-win_amd64.whl.metadata (60 kB)
Collecting diskcache>=5.6.1 (from llama-cpp-python)
  Downloading diskcache-5.6.3-py3-none-any.whl.metadata (20 kB)
Collecting jinja2>=2.11.3 (from llama-cpp-python)
  Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting MarkupSafe>=2.0 (from jinja2>=2.11.3->llama-cpp-python)
  Downloading MarkupSafe-3.0.2-cp312-cp312-win_amd64.whl.metadata (4.1 kB)
Downloading diskcache-5.6.3-py3-none-any.whl (45 kB)
Downloading jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading MarkupSafe-3.0.2-cp312-cp312-win_amd64.whl (15 kB)
Downloading numpy-2.2.6-cp312-cp312-win_amd64.whl (12.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.6/12.6 MB 41.6 MB/s eta 0:00:00
Downloading typing_extensions-4.14.0-py3-none-any.whl (43 kB)
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml) ... done
  Created wheel for llama-cpp-python: filename=llama_cpp_python-0.3.9-cp312-cp312-win_amd64.whl size=3018273 sha256=598c51298accb9571e41a46494f887bbe6354e4eb4beb75bf3b957e00b7ac84d
  Stored in directory: C:\Users\Peter\AppData\Local\Temp\pip-ephem-wheel-cache-dzid_72r\wheels\e9\22\42\98dca29f6195951fae2aa548582827a45306350e282ab30617
Successfully built llama-cpp-python
Installing collected packages: typing-extensions, numpy, MarkupSafe, diskcache, jinja2, llama-cpp-python
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.14.0
    Uninstalling typing_extensions-4.14.0:
      Successfully uninstalled typing_extensions-4.14.0
  Attempting uninstall: numpy
    Found existing installation: numpy 2.2.6
    Uninstalling numpy-2.2.6:
      Successfully uninstalled numpy-2.2.6
  Attempting uninstall: MarkupSafe
    Found existing installation: MarkupSafe 3.0.2
    Uninstalling MarkupSafe-3.0.2:
      Successfully uninstalled MarkupSafe-3.0.2
  Attempting uninstall: diskcache
    Found existing installation: diskcache 5.6.3
    Uninstalling diskcache-5.6.3:
      Successfully uninstalled diskcache-5.6.3
  Attempting uninstall: jinja2
    Found existing installation: Jinja2 3.1.6
    Uninstalling Jinja2-3.1.6:
      Successfully uninstalled Jinja2-3.1.6
  Attempting uninstall: llama-cpp-python
    Found existing installation: llama_cpp_python 0.3.4
    Uninstalling llama_cpp_python-0.3.4:
      Successfully uninstalled llama_cpp_python-0.3.4
Successfully installed MarkupSafe-3.0.2 diskcache-5.6.3 jinja2-3.1.6 llama-cpp-python-0.3.9 numpy-2.2.6 typing-extensions-4.14.0
