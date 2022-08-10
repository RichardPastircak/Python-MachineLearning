<h1>Python Machine Learning projekt</h1>
V rámci projektu som začal spracovaním datasetu X do vhodnej podoby na trénovanie. V tomto kroku som zároveň aj pripravil dataset Y na budúcu predikciu. 
Následne som na datasete X prostredníctvom gridsearchu natrénoval niekoľko classifikátorov ( __SVC__ ,**DTC**,**RFC**,**nuSVC**).Vybral som tej s najväčšou 
prestnosťou, natrénoval ho na dátach z Y a predikoval na ňom.

<h2>Súbory</h2>

  - X_public: x-ová os datasetu slúžiaceho na učenie,
  - X_eval: y-ová os datasetu slúžiaceho na učenie,
  - y_public: x-ová os datasetu slúžiaceho na predikciu,
  - y_eval: mnou zostrojená predikcia
