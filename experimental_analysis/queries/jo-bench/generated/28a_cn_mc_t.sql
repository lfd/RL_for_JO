SELECT * FROM movie_companies AS mc, title AS t, company_name AS cn WHERE cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2000 AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id;