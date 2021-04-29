import os
import lwsspy as lpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


if __name__ == "__main__":

    # River numbers

    stations = {
        "Lucas Sawade": '06817500',
        "Matthew Butler": '4127997',
        "William Carpenter": '1199000',
        "Kaila Carroll": '1205500',
        "Shannon Chaffers": '4174500',
        "Sophia Duchateau": '50075000',
        "Michael Folding": '1208990',
        "Lauren Huff": '2089500',
        "Caren Ju": '4102500',
        "Sarah Kamanzi": '1175670',
        "Elias Mosby": '8062000',
        "Jay Rolader": '2336300',
        "Isabel Segel": '3085500',
        "Yuxin Shi": '11023340',
        "Isra Thange": '4074950'
    }


# Get Stationname
with PdfPages(os.path.join(lpy.DOCFIGURES, f"rivers.pdf")) as pdf:
    for _student, _station in stations.items():
        print(_station)
        try:
            r = lpy.River(_station, pre_title=_student, save=False)
            r.populate()
            r.plot_summary()
            # r.plot_stage_v_discharge_evo()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
        except Exception as e:
            print(e)
