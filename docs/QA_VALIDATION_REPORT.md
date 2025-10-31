E
==================================== ERRORS ====================================
_____________________ ERROR at setup of test_api_endpoint ______________________
file /home/mo/thunderline/cerebros-core-algorithm-alpha/test_e2e.py, line 42
  def test_api_endpoint(method, endpoint, description, data=None):
E       fixture 'method' not found
>       available fixtures: anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, doctest_namespace, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, monkeypatch, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.

/home/mo/thunderline/cerebros-core-algorithm-alpha/test_e2e.py:42
=========================== short test summary info ============================
ERROR test_e2e.py::test_api_endpoint
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 error in 0.19s
