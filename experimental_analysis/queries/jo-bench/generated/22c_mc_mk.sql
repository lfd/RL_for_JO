SELECT * FROM movie_keyword AS mk, movie_companies AS mc WHERE mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id;