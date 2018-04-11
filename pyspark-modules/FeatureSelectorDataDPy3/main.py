
from datacanvas.new_runtime import DataCanvas
dc = DataCanvas(__name__)

@dc.basic_runtime(spec_json="spec.json")
def main(rt, params, inputs, outputs):
    from run import main
    main(params, inputs, outputs)
    print("done")

if __name__ == "__main__":
    dc.run()

         