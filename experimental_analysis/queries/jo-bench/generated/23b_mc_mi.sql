SELECT * FROM movie_companies AS mc, movie_info AS mi WHERE mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id;