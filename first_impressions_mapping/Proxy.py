from Logger import Logger

# The purpose of this proxy is to allow for a dynamic proxy interception of any generic class to catch method calls between the Orchestrator and 
# supporting classes. This allows for run-time logging without explicitly calling the logging class, along with other cool 
# overrides that will allow for changing parameters to fit that of passed models dynamically at run-time if needed.
#see https://stackoverflow.com/questions/1466676/create-a-wrapper-class-to-call-a-pre-and-post-function-around-existing-functions
class Proxy(object):

    def __init__(self, proxied_class):

        self.proxied_class = proxied_class
        self.Logger = Logger('./logs/', 'log')

    def __getattr__(self, attr):

        orig_attr = self.proxied_class.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):

                self.pre(attr) # used for after proxy call logging

                try:
                    result = orig_attr(*args, **kwargs)
                except Exception as e:
                    self.Logger.Error("Error occured while running method: " + str(e))
                    return

                # prevent proxied_class from becoming unwrapped
                if result == self.proxied_class:
                    return self

                self.post(result) # used for after proxy calls 
                return result
            return hooked
        else:
            return orig_attr

    def pre(self, attr):
        self.Logger.Info("Calling method: " + str(attr))

    def post(self, value):
        data = str(value)
        info = (data[:75] + '...') if len(data) > 200 else data
        self.Logger.Info("Method returned: " + info)
        